import argparse
import copy
import multiprocessing
import os
import warnings
import atexit
import openpifpaf
import psutil
import signal
import torch.multiprocessing as mp


from algorithms import *


def handle_exit(*args):
    try:
        logging.info("Exiting-handle_exit")

        for p in psutil.process_iter():
            if p.name() == "fall_detector.exe":
                logging.info(
                    "{} this_app usage CPU={} RAM={}".format(time.time(), p.cpu_percent(), p.memory_percent()))

        logging.info("{} Fall alert handling exit cpu={}\nmemory={}% {}".format(time.time(),
                                                                                psutil.cpu_percent(
                                                                                    interval=0.200000000001,
                                                                                    percpu=True),
                                                                                psutil.virtual_memory().available * 100 / psutil.virtual_memory().total,
                                                                                dict(
                                                                                    psutil.virtual_memory()._asdict())))

    except Exception as e:
        print("handle_exit with error", e)


atexit.register(handle_exit)
signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT, handle_exit)

t = time.time()
if not os.path.exists("logs"):
    os.mkdir("logs")
    logging.basicConfig(
        filename="logs/logfile.{}..{}.log".format(datetime.datetime.now().strftime('%m-%d %H-%M-%S'), t),
        level=logging.INFO)
    logging.info("*\n" * 5 + "{} Started the program".format(time.time()))

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass


class FallDetector:
    def __init__(self, t=DEFAULT_CONSEC_FRAMES):
        self.consecutive_frames = t
        self.args = self.cli()

    def cli(self):
        parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        openpifpaf.network.Factory.cli(parser)
        openpifpaf.decoder.cli(parser)
        parser.add_argument('--resolution', default=0.4, type=float,
                            help=('Resolution prescale factor from 640x480. '
                                  'Will be rounded to multiples of 16.'))
        parser.add_argument('--resize', default=None, type=str,
                            help=('Force input image resize. '
                                  'Example WIDTHxHEIGHT.'))
        parser.add_argument('--num_cams', default=1, type=int,
                            help='Number of Cameras.')
        parser.add_argument('--video', default=None, type=str,
                            help='Path to the video file.\nFor single video fall detection(--num_cams=1), save your videos as abc.xyz and set --video=abc.xyz\nFor 2 video fall detection(--num_cams=2), save your videos as abc1.xyz & abc2.xyz and set --video=abc.xyz')
        parser.add_argument('--debug', default=False, action='store_true',
                            help='debug messages and autoreload')
        parser.add_argument('--disable_cuda', default=False, action='store_true',
                            help='disables cuda support and runs from gpu')

        vis_args = parser.add_argument_group('Visualisation')
        vis_args.add_argument('--plot_graph', default=False, action='store_true',
                              help='Plot the graph of features extracted from keypoints of pose.')
        vis_args.add_argument('--joints', default=True, action='store_true',
                              help='Draw joint\'s keypoints on the output video.')
        vis_args.add_argument('--skeleton', default=True, action='store_true',
                              help='Draw skeleton on the output video.')
        vis_args.add_argument('--coco_points', default=False, action='store_true',
                              help='Visualises the COCO points of the human pose.')
        vis_args.add_argument('--save_output', default=False, action='store_true',
                              help='Save the result in a video file. Output videos are saved in the same directory as input videos with "out" appended at the start of the title')
        vis_args.add_argument('--fps', default=18, type=int,
                              help='FPS for the output video.')

        args = parser.parse_args()
        args.force_complete_pose = True
        args.instance_threshold = 0.2
        args.seed_threshold = 0.5

        args.device = torch.device('cpu')
        args.pin_memory = False
        if not args.disable_cuda and torch.cuda.is_available():
            args.device = torch.device('cuda')
            args.pin_memory = True

        if args.checkpoint is None:
            args.checkpoint = 'shufflenetv2k16w'

        openpifpaf.decoder.configure(args)
        openpifpaf.network.Factory.configure(args)

        return args

    def begin(self):
        print('Starting...')
        logging.info("{} Starting... - FallDetector begined".format(time.time()))
        e = mp.Event()
        queues = [mp.Queue() for _ in range(self.args.num_cams)]
        counter1 = mp.Value('i', 0)
        counter2 = mp.Value('i', 0)
        argss = [copy.deepcopy(self.args) for _ in range(self.args.num_cams)]
        if self.args.num_cams == 1:
            if self.args.video is None:
                argss[0].video = 0
            process1 = mp.Process(target=extract_keypoints_parallel,
                                  args=(queues[0], argss[0], counter1, counter2, self.consecutive_frames, e))
            process1.start()
            if self.args.coco_points:
                process1.join()
            else:
                process2 = mp.Process(target=alg2_sequential, args=(queues, argss,
                                                                    self.consecutive_frames, e))
                process2.start()
            process1.join()
            logging.info("{} joined process".format(time.time()))
        elif self.args.num_cams == 2:
            if self.args.video is None:
                argss[0].video = 0
                argss[1].video = 1
            else:
                try:
                    vid_name = self.args.video.split('.')
                    argss[0].video = ''.join(vid_name[:-1]) + '1.' + vid_name[-1]
                    argss[1].video = ''.join(vid_name[:-1]) + '2.' + vid_name[-1]
                    print('Video 1:', argss[0].video)
                    print('Video 2:', argss[1].video)
                except Exception as exep:
                    print('Error: argument --video not properly set')
                    return

            logging.info("Starting processes")
            process1_1 = mp.Process(target=extract_keypoints_parallel,
                                    args=(queues[0], argss[0], counter1, counter2, self.consecutive_frames, e))
            process1_2 = mp.Process(target=extract_keypoints_parallel,
                                    args=(queues[1], argss[1], counter2, counter1, self.consecutive_frames, e))
            process1_1.start()
            process1_2.start()
            if self.args.coco_points:
                process1_1.join()
                process1_2.join()
            else:
                process2 = mp.Process(target=alg2_sequential, args=(queues, argss,
                                                                    self.consecutive_frames, e))
                process2.start()
            process1_1.join()
            process1_2.join()
            logging.info("{} joined processes".format(time.time()))
        else:
            print('More than 2 cameras are currently not supported')
            return

        if not self.args.coco_points:
            process2.join()
        print(f'{time.time()} Exiting...')
        return


if __name__ == "__main__":
    multiprocessing.freeze_support()
    warnings.filterwarnings("ignore")

    logging.info("{} Starting FallDetector".format(time.time()))
    f = FallDetector()
    logging.info("{} Initialized FallDetector".format(time.time()))
    f.begin()
    logging.info("{} Program Ended Successfully".format(time.time()))
