import sys
import csv

sys.path.extend([".", ".."])
from CONSTANTS import *
from sklearn.decomposition import FastICA
from representations.templates.statistics import Simple_template_TF_IDF, Template_TF_IDF_without_clean
from representations.sequences.statistics import Sequential_TF
from preprocessing.datacutter.SimpleCutting import cut_by_613
from preprocessing.AutoLabeling import Probabilistic_Labeling
from preprocessing.Preprocess import Preprocessor
from module.Optimizer import Optimizer
from module.Common import data_iter, generate_tinsts_binary_label, batch_variable_inst
from models.gru import AttGRUModel
from utils.Vocab import Vocab
from entities.instances import Instance
from parsers.Drain_IBM import Drain3Parser

lstm_hiddens = 100
num_layer = 2
batch_size = 100
epochs = 5


def _remove_columns(line, remove_cols):
    tokens = line.strip().split()
    return ' '.join([token for idx, token in enumerate(tokens) if idx not in remove_cols])


def preprocess_line_for_dataset(dataset, line):
    dataset_lower = dataset.lower()
    if dataset_lower == 'hdfs':
        return _remove_columns(line, [0, 1, 2, 3, 4])
    elif dataset_lower in ('bgl', 'bglsample'):
        return _remove_columns(line, list(range(9)))
    else:
        return line.rstrip('\n')


def load_template_embeddings(embedding_file):
    id2embed = {}
    if not os.path.exists(embedding_file):
        return id2embed
    with open(embedding_file, 'r', encoding='utf-8') as reader:
        for line in reader:
            tokens = line.strip().split()
            if len(tokens) <= 1:
                continue
            try:
                temp_id = int(tokens[0])
            except ValueError:
                continue
            vector = np.array([float(x) for x in tokens[1:]], dtype=np.float64)
            id2embed[temp_id] = vector
    return id2embed


def resolve_parser_config(dataset):
    dataset_lower = dataset.lower()
    if dataset_lower == 'hdfs':
        return os.path.join(PROJECT_ROOT, 'conf/HDFS.ini')
    if dataset_lower in ('bgl', 'bglsample'):
        return os.path.join(PROJECT_ROOT, 'conf/BGL.ini')
    if dataset_lower == 'open5gs':
        return os.path.join(PROJECT_ROOT, 'conf/HDFS.ini')
    return None


def run_inference(dataset, parser_name, inference_file, parser_config, parser_persistence,
                  output_model_dir, threshold):
    logger = logging.getLogger('PLELog')
    if not inference_file:
        logger.error('No inference file provided.')
        return
    if not os.path.exists(inference_file):
        logger.error('Inference file %s not found.', inference_file)
        return
    if parser_config is None:
        logger.error('No parser configuration available for dataset %s.', dataset)
        return

    drain_parser = Drain3Parser(config_file=parser_config, persistence_folder=parser_persistence)
    if drain_parser.to_update:
        logger.error('No trained parser state found for dataset %s. Please run training first.', dataset)
        return

    templates_embedding_file = os.path.join(drain_parser.persistence_folder, 'templates.vec')
    id2embed = load_template_embeddings(templates_embedding_file)
    if not id2embed:
        logger.error('Failed to load template embeddings from %s.', templates_embedding_file)
        return

    vocab = Vocab()
    vocab.load_from_dict(id2embed)
    label2id = {'Normal': 0, 'Anomalous': 1}
    plelog = PLELog(vocab, num_layer, lstm_hiddens, label2id)

    log = 'layer={}_hidden={}_epoch={}'.format(num_layer, lstm_hiddens, epochs)
    best_model_file = os.path.join(output_model_dir, log + '_best.pt')
    last_model_file = os.path.join(output_model_dir, log + '_last.pt')
    model_path = None
    if os.path.exists(best_model_file):
        model_path = best_model_file
    elif os.path.exists(last_model_file):
        model_path = last_model_file
    if model_path is None:
        logger.error('No trained model weights found under %s. Please train the model first.', output_model_dir)
        return

    plelog.model.load_state_dict(torch.load(model_path, map_location=device))
    plelog.logger.info('Loaded model weights from %s', model_path)
    plelog.model.eval()

    id2tag = {v: k for k, v in label2id.items()}
    inference_instances = []
    id_to_message = {}
    ordered_ids = []

    with open(inference_file, 'r', encoding='utf-8', errors='ignore') as reader:
        for line_idx, raw_line in enumerate(reader):
            raw_message = raw_line.rstrip('\n')
            processed_line = preprocess_line_for_dataset(dataset, raw_line)
            cluster = drain_parser.match(processed_line)
            if cluster is not None:
                try:
                    event_id = int(cluster.cluster_id)
                except (TypeError, ValueError):
                    event_id = -1
            else:
                event_id = -1

            block_id = f"{os.path.basename(inference_file)}:{line_idx}"
            inst = Instance(block_id, [event_id], 'Normal')
            inference_instances.append(inst)
            id_to_message[block_id] = raw_message
            ordered_ids.append(block_id)

    output_dir = os.path.join(PROJECT_ROOT, 'outputs',
                               'results/PLELog/' + dataset + '_' + parser_name + '/inference')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir,
                               os.path.splitext(os.path.basename(inference_file))[0] + '_predictions.csv')

    if not inference_instances:
        logger.warning('Inference file %s is empty. Generated an empty result file at %s.', inference_file, output_file)
        with open(output_file, 'w', encoding='utf-8', newline='') as writer:
            csv_writer = csv.writer(writer)
            csv_writer.writerow(['line_number', 'block_id', 'log', 'predicted_label', 'anomaly_score'])
        return

    results = {}
    anomaly_index = label2id['Anomalous']

    with torch.no_grad():
        plelog.model.eval()
        for onebatch in data_iter(inference_instances, plelog.test_batch_size, False):
            tinst = generate_tinsts_binary_label(onebatch, vocab, False)
            tinst.to_cuda(device)
            pred_tags, tag_logits = plelog.predict(tinst.inputs, threshold if threshold is not None else None)
            logits_np = tag_logits.detach().cpu().numpy()
            if isinstance(pred_tags, torch.Tensor):
                preds_np = pred_tags.detach().cpu().numpy()
            else:
                preds_np = pred_tags
            for inst, pred, prob in zip(onebatch, preds_np, logits_np):
                results[inst.id] = {'predicted_label': id2tag[int(pred)],
                                    'anomaly_score': float(prob[anomaly_index])}

    with open(output_file, 'w', encoding='utf-8', newline='') as writer:
        csv_writer = csv.writer(writer)
        csv_writer.writerow(['line_number', 'block_id', 'log', 'predicted_label', 'anomaly_score'])
        for index, block_id in enumerate(ordered_ids):
            record = results.get(block_id, {'predicted_label': 'Normal', 'anomaly_score': 0.0})
            csv_writer.writerow([index, block_id, id_to_message.get(block_id, ''),
                                 record['predicted_label'],
                                 '{:.6f}'.format(record['anomaly_score'])])

    logger.info('Inference results saved to %s', output_file)


class PLELog:
    _logger = logging.getLogger('PLELog')
    _logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))
    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'PLELog.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))
    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)
    _logger.info(
        'Construct logger for PLELog succeeded, current working directory: %s, logs will be written in %s' %
        (os.getcwd(), LOG_ROOT))

    @property
    def logger(self):
        return PLELog._logger

    def __init__(self, vocab, num_layer, hidden_size, label2id):
        self.label2id = label2id
        self.vocab = vocab
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.batch_size = 128
        self.test_batch_size = 1024
        self.model = AttGRUModel(vocab, self.num_layer, self.hidden_size)
        if torch.cuda.is_available():
            self.model = self.model.cuda(device)
        self.loss = nn.BCELoss()

    def forward(self, inputs, targets):
        tag_logits = self.model(inputs)
        tag_logits = F.softmax(tag_logits, dim=1)
        loss = self.loss(tag_logits, targets)
        return loss

    def predict(self, inputs, threshold=None):
        with torch.no_grad():
            tag_logits = self.model(inputs)
            tag_logits = F.softmax(tag_logits)
        if threshold is not None:
            probs = tag_logits.detach().cpu().numpy()
            anomaly_id = self.label2id['Anomalous']
            pred_tags = np.zeros(probs.shape[0])
            for i, logits in enumerate(probs):
                if logits[anomaly_id] >= threshold:
                    pred_tags[i] = anomaly_id
                else:
                    pred_tags[i] = 1 - anomaly_id

        else:
            pred_tags = tag_logits.detach().max(1)[1].cpu()
        return pred_tags, tag_logits

    def evaluate(self, instances, threshold=0.5):
        self.logger.info('Start evaluating by threshold %.3f' % threshold)
        with torch.no_grad():
            self.model.eval()
            globalBatchNum = 0
            TP, TN, FP, FN = 0, 0, 0, 0
            tag_correct, tag_total = 0, 0
            for onebatch in data_iter(instances, self.test_batch_size, False):
                tinst = generate_tinsts_binary_label(onebatch, vocab, False)
                tinst.to_cuda(device)
                self.model.eval()
                pred_tags, tag_logits = self.predict(tinst.inputs, threshold)
                for inst, bmatch in batch_variable_inst(onebatch, pred_tags, tag_logits, processor.id2tag):
                    tag_total += 1
                    if bmatch:
                        tag_correct += 1
                        if inst.label == 'Normal':
                            TN += 1
                        else:
                            TP += 1
                    else:
                        if inst.label == 'Normal':
                            FP += 1
                        else:
                            FN += 1
                globalBatchNum += 1
            self.logger.info('TP: %d, TN: %d, FN: %d, FP: %d' % (TP, TN, FN, FP))
            if TP + FP != 0:
                precision = 100 * TP / (TP + FP)
                recall = 100 * TP / (TP + FN)
                f = 2 * precision * recall / (precision + recall)
                end = time.time()
                self.logger.info('Precision = %d / %d = %.4f, Recall = %d / %d = %.4f F1 score = %.4f'
                                 % (TP, (TP + FP), precision, TP, (TP + FN), recall, f))
            else:
                self.logger.info('Precision is 0 and therefore f is 0')
                precision, recall, f = 0, 0, 0
        return precision, recall, f


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', default='HDFS', type=str, help='Target dataset. Default HDFS')
    argparser.add_argument('--mode', default='train', type=str, help='train, test or inference')
    argparser.add_argument('--parser', default='IBM', type=str,
                           help='Select parser, please see parser list for detail. Default Official.')
    argparser.add_argument('--min_cluster_size', type=int, default=100,
                           help="min_cluster_size.")
    argparser.add_argument('--min_samples', type=int, default=100,
                           help="min_samples")
    argparser.add_argument('--reduce_dimension', type=int, default=50,
                           help="Reduce dimentsion for fastICA, to accelerate the HDBSCAN probabilistic label estimation.")
    argparser.add_argument('--threshold', type=float, default=0.5,
                           help="Anomaly threshold for PLELog.")
    argparser.add_argument('--inference_file', type=str, default=None,
                           help='Run inference on the specified log file using a trained model.')
    args, extra_args = argparser.parse_known_args()

    dataset = args.dataset
    parser = args.parser
    mode = args.mode
    min_cluster_size = args.min_cluster_size
    min_samples = args.min_samples
    reduce_dimension = args.reduce_dimension
    threshold = args.threshold
    inference_file = args.inference_file

    # Mark results saving directories.
    save_dir = os.path.join(PROJECT_ROOT, 'outputs')
    base = os.path.join(PROJECT_ROOT, 'datasets/' + dataset)
    output_model_dir = os.path.join(save_dir, 'models/PLELog/' + dataset + '_' + parser + '/model')
    output_res_dir = os.path.join(save_dir, 'results/PLELog/' + dataset + '_' + parser + '/detect_res')
    prob_label_res_file = os.path.join(save_dir,
                                       'results/PLELog/' + dataset + '_' + parser +
                                       '/prob_label_res/mcs-' + str(min_cluster_size) + '_ms-' + str(min_samples))
    rand_state = os.path.join(save_dir,
                              'results/PLELog/' + dataset + '_' + parser +
                              '/prob_label_res/random_state')
    save_dir = os.path.join(PROJECT_ROOT, 'outputs')
    parser_config = resolve_parser_config(dataset)
    parser_persistence = os.path.join(PROJECT_ROOT, 'datasets/' + dataset + '/persistences')

    if inference_file:
        run_inference(dataset, parser, inference_file, parser_config, parser_persistence,
                      output_model_dir, threshold)
        sys.exit(0)

    # Training, Validating and Testing instances.
    template_encoder = Template_TF_IDF_without_clean() if dataset == 'NC' else Simple_template_TF_IDF()
    processor = Preprocessor()
    train, dev, test = processor.process(dataset=dataset, parsing=parser, cut_func=cut_by_613,
                                         template_encoding=template_encoder.present)
    num_classes = len(processor.train_event2idx)

    # Log sequence representation.
    sequential_encoder = Sequential_TF(processor.embedding)
    train_reprs = sequential_encoder.present(train)
    for index, inst in enumerate(train):
        inst.repr = train_reprs[index]
    # dev_reprs = sequential_encoder.present(dev)
    # for index, inst in enumerate(dev):
    #     inst.repr = dev_reprs[index]
    test_reprs = sequential_encoder.present(test)
    for index, inst in enumerate(test):
        inst.repr = test_reprs[index]

    # Dimension reduction if specified.
    transformer = None
    if reduce_dimension != -1:
        start_time = time.time()
        print("Start FastICA, target dimension: %d" % reduce_dimension)
        transformer = FastICA(n_components=reduce_dimension)
        train_reprs = transformer.fit_transform(train_reprs)
        for idx, inst in enumerate(train):
            inst.repr = train_reprs[idx]
        print('Finished at %.2f' % (time.time() - start_time))

    # Probabilistic labeling.
    # Sample normal instances.
    train_normal = [x for x, inst in enumerate(train) if inst.label == 'Normal']
    normal_ids = train_normal[:int(0.5 * len(train_normal))]
    label_generator = Probabilistic_Labeling(min_samples=min_samples, min_clust_size=min_cluster_size,
                                             res_file=prob_label_res_file, rand_state_file=rand_state)

    labeled_train = label_generator.auto_label(train, normal_ids)

    # Below is used to test if the loaded result match the original clustering result.
    TP, TN, FP, FN = 0, 0, 0, 0

    for inst in labeled_train:
        if inst.predicted == 'Normal':
            if inst.label == 'Normal':
                TN += 1
            else:
                FN += 1
        else:
            if inst.label == 'Anomalous':
                TP += 1
            else:
                FP += 1
    from utils.common import get_precision_recall

    print(len(normal_ids))
    print('TP %d TN %d FP %d FN %d' % (TP, TN, FP, FN))
    p, r, f = get_precision_recall(TP, TN, FP, FN)
    print('%.4f, %.4f, %.4f' % (p, r, f))

    # Load Embeddings
    vocab = Vocab()
    vocab.load_from_dict(processor.embedding)

    plelog = PLELog(vocab, num_layer, lstm_hiddens, processor.label2id)

    log = 'layer={}_hidden={}_epoch={}'.format(num_layer, lstm_hiddens, epochs)
    best_model_file = os.path.join(output_model_dir, log + '_best.pt')
    last_model_file = os.path.join(output_model_dir, log + '_last.pt')
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)
    if mode == 'train':
        # Train
        optimizer = Optimizer(filter(lambda p: p.requires_grad, plelog.model.parameters()))
        bestClassifier = None
        global_step = 0
        bestF = 0
        batch_num = int(np.ceil(len(labeled_train) / float(batch_size)))

        for epoch in range(epochs):
            plelog.model.train()
            start = time.strftime("%H:%M:%S")
            plelog.logger.info("Starting epoch: %d | phase: train | start time: %s | learning rate: %s" %
                               (epoch + 1, start, optimizer.lr))
            batch_iter = 0
            correct_num, total_num = 0, 0
            # start batch
            for onebatch in data_iter(labeled_train, batch_size, True):
                plelog.model.train()
                tinst = generate_tinsts_binary_label(onebatch, vocab)
                tinst.to_cuda(device)
                loss = plelog.forward(tinst.inputs, tinst.targets)
                loss_value = loss.data.cpu().numpy()
                loss.backward()
                if batch_iter % 100 == 0:
                    plelog.logger.info("Step:%d, Iter:%d, batch:%d, loss:%.2f" \
                                       % (global_step, epoch, batch_iter, loss_value))
                batch_iter += 1
                if batch_iter % 1 == 0 or batch_iter == batch_num:
                    nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, plelog.model.parameters()),
                        max_norm=1)
                    optimizer.step()
                    plelog.model.zero_grad()
                    global_step += 1
                if dev:
                    if batch_iter % 500 == 0 or batch_iter == batch_num:
                        plelog.logger.info('Testing on test set.')
                        _, _, f = plelog.evaluate(dev)
                        if f > bestF:
                            plelog.logger.info("Exceed best f: history = %.2f, current = %.2f" % (bestF, f))
                            torch.save(plelog.model.state_dict(), best_model_file)
                            bestF = f
            plelog.logger.info('Training epoch %d finished.' % epoch)
            torch.save(plelog.model.state_dict(), last_model_file)

    if os.path.exists(last_model_file):
        plelog.logger.info('=== Final Model ===')
        plelog.model.load_state_dict(torch.load(last_model_file))
        plelog.evaluate(test, threshold)
    if os.path.exists(best_model_file):
        plelog.logger.info('=== Best Model ===')
        plelog.model.load_state_dict(torch.load(best_model_file))
        plelog.evaluate(test, threshold)
    plelog.logger.info('All Finished')
