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
        log_paths = []
        for root, _, files in os.walk(self.logs_root):
            for filename in sorted(files):
                if filename.endswith('.log'):
                    log_paths.append(os.path.join(root, filename))
        log_paths.sort()

        if not log_paths:
            self.logger.error('No .log files found under %s. Please verify the dataset contents.', self.logs_root)
            exit(1)

        total_lines = 0
        self.blocks = []
        self.block2seqs = {}
        self.block2label = {}

        with open(self.in_file, 'w', encoding='utf-8') as writer:
            for path in log_paths:
                relative_path = os.path.relpath(path, self.logs_root)
                with open(path, 'r', encoding='utf-8', errors='ignore') as reader:
                    for line_idx, raw_line in enumerate(reader):
                        message = raw_line.rstrip('\n')
                        writer.write(message + '\n')
                        block_id = f"{relative_path}:{line_idx}"
                        self.blocks.append(block_id)
                        self.block2seqs[block_id] = [total_lines]
                        self.block2label[block_id] = 'Normal'
                        total_lines += 1

        self.logger.info('Collected %d log lines from %d files under %s.', total_lines, len(log_paths), self.logs_root)
