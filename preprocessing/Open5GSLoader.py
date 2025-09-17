import sys

sys.path.extend([".", ".."])  # noqa: E402
from CONSTANTS import *  # noqa: E402
from preprocessing.BasicLoader import BasicDataLoader  # noqa: E402


class Open5GSLoader(BasicDataLoader):
    def __init__(self, logs_root=os.path.join(PROJECT_ROOT, 'datasets/open5gs/logs'),
                 dataset_base=os.path.join(PROJECT_ROOT, 'datasets/open5gs'),
                 semantic_repr_func=None):
        super(Open5GSLoader, self).__init__()

        self.logger = logging.getLogger('Open5GSLoader')
        self.logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

        file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'Open5GSLoader.log'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.info(
            'Construct Open5GSLoader logger success, current working directory: %s, logs will be written in %s',
            os.getcwd(), LOG_ROOT)

        if logs_root is None:
            logs_root = os.path.join(PROJECT_ROOT, 'datasets/open5gs/logs')
        if dataset_base is None:
            dataset_base = os.path.join(PROJECT_ROOT, 'datasets/open5gs')

        if not os.path.exists(logs_root):
            self.logger.error('Log root %s not found, please check the dataset path.', logs_root)
            exit(1)

        self.logs_root = logs_root
        self.dataset_base = dataset_base
        if not os.path.exists(self.dataset_base):
            os.makedirs(self.dataset_base)

        self.in_file = os.path.join(self.dataset_base, 'open5gs_combined.log')
        self.remove_cols = []
        self.semantic_repr_func = semantic_repr_func
        self._load_raw_log_seqs()

    def logger(self):
        return self.logger

    def _pre_process(self, line):
        return line.rstrip('\n')

    def _load_raw_log_seqs(self):
        candidate_clean_root = os.path.join(self.logs_root, 'clean')
        if os.path.isdir(candidate_clean_root):
            normal_root = candidate_clean_root
        else:
            normal_root = self.logs_root

        normal_files = self._collect_log_files(normal_root)
        if not normal_files:
            self.logger.error('No normal log files found under %s. Please verify the dataset structure.', normal_root)
            exit(1)

        anomalous_root = os.path.join(self.dataset_base, 'anomalous_logs')
        anomalous_files = self._collect_log_files(anomalous_root) if os.path.isdir(anomalous_root) else []
        if not anomalous_files:
            self.logger.warning('No anomalous log files were found under %s. The unlabeled set will only contain '
                                'Normal samples.', anomalous_root)

        total_lines = 0
        self.blocks = []
        self.block2seqs = {}
        self.block2label = {}
        self.block2labeled = {}
        self.block2origin = {}

        normal_entries = []
        anomalous_entries = []

        with open(self.in_file, 'w', encoding='utf-8') as writer:
            for path in normal_files:
                rel_path = os.path.relpath(path, normal_root)
                normalized_rel_path = rel_path.replace(os.sep, '/')
                line_ids = []
                with open(path, 'r', encoding='utf-8', errors='ignore') as reader:
                    for raw_line in reader:
                        message = self._pre_process(raw_line)
                        writer.write(message + '\n')
                        line_ids.append(total_lines)
                        total_lines += 1
                if not line_ids:
                    self.logger.warning('Skipped empty normal log file %s', path)
                    continue
                normal_entries.append((normalized_rel_path, line_ids))

            for path in anomalous_files:
                rel_path = os.path.relpath(path, anomalous_root)
                normalized_rel_path = rel_path.replace(os.sep, '/')
                line_ids = []
                with open(path, 'r', encoding='utf-8', errors='ignore') as reader:
                    for raw_line in reader:
                        message = self._pre_process(raw_line)
                        writer.write(message + '\n')
                        line_ids.append(total_lines)
                        total_lines += 1
                if not line_ids:
                    self.logger.warning('Skipped empty anomalous log file %s', path)
                    continue
                anomalous_entries.append((normalized_rel_path, line_ids))

        labeled_block_count = 0
        for rel_path, line_ids in normal_entries:
            block_id = f"normal_labeled::{rel_path}"
            self.blocks.append(block_id)
            self.block2seqs[block_id] = list(line_ids)
            self.block2label[block_id] = 'Normal'
            self.block2labeled[block_id] = True
            self.block2origin[block_id] = 'labeled_normal'
            labeled_block_count += 1

        unlabeled_normal_blocks = []
        for rel_path, line_ids in normal_entries:
            block_id = f"normal_unlabeled::{rel_path}"
            self.block2seqs[block_id] = list(line_ids)
            self.block2label[block_id] = 'Normal'
            self.block2labeled[block_id] = False
            self.block2origin[block_id] = 'unlabeled_normal'
            unlabeled_normal_blocks.append(block_id)

        unlabeled_anomalous_blocks = []
        for rel_path, line_ids in anomalous_entries:
            block_id = f"anomalous_unlabeled::{rel_path}"
            self.block2seqs[block_id] = list(line_ids)
            self.block2label[block_id] = 'Anomalous'
            self.block2labeled[block_id] = False
            self.block2origin[block_id] = 'unlabeled_anomalous'
            unlabeled_anomalous_blocks.append(block_id)

        mixed_unlabeled_blocks = []
        max_len = max(len(unlabeled_normal_blocks), len(unlabeled_anomalous_blocks))
        for idx in range(max_len):
            if idx < len(unlabeled_normal_blocks):
                mixed_unlabeled_blocks.append(unlabeled_normal_blocks[idx])
            if idx < len(unlabeled_anomalous_blocks):
                mixed_unlabeled_blocks.append(unlabeled_anomalous_blocks[idx])

        self.blocks.extend(mixed_unlabeled_blocks)

        self.logger.info('Collected %d log lines from %d normal files and %d anomalous files.',
                         total_lines, len(normal_entries), len(anomalous_entries))
        self.logger.info('Prepared %d labeled Normal sequences and %d unlabeled sequences (Normal=%d, Anomalous=%d).',
                         labeled_block_count, len(mixed_unlabeled_blocks),
                         len(unlabeled_normal_blocks), len(unlabeled_anomalous_blocks))

    def _collect_log_files(self, root):
        log_files = []
        if not root or not os.path.exists(root):
            return log_files
        for current_root, _, files in os.walk(root):
            for filename in sorted(files):
                if filename.endswith('.log'):
                    log_files.append(os.path.join(current_root, filename))
        log_files.sort()
        return log_files
