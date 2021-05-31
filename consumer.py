import sys
import argparse
import time
from kafka import KafkaConsumer
import utils
import pickle
from threading import Thread
from collections import deque
from predict import *

INTERVAL = 1


class MyConsumer(object):
    def __init__(self, args):
        self.args = args
        self.latency_samples = []
        self.e2e_latency_samples = []
        self.throughput_samples = []
        self.last_timestamp = None

        self.model = load_model()
        self.img_queue = deque(maxlen=100)

    def capture_metrics(self, consumer_metrics, recv_time, send_time):
        try:
            metrics = consumer_metrics['consumer-metrics']
            _latency = metrics['request-latency-avg']
            _throughput = metrics['incoming-byte-rate']
            _e2e_latency = 1000.0 * recv_time - send_time
            self.latency_samples.append(_latency)
            self.throughput_samples.append(_throughput)
            self.e2e_latency_samples.append(_e2e_latency)
        except Exception:
            pass
        finally:
            self.last_timestamp = time.time()

    def consume_msg(self):
        consumer = KafkaConsumer(self.args.topic,
                                 client_id=self.args.client_id,
                                 group_id='group-1',
                                 auto_offset_reset='latest',
                                 metrics_num_samples=100,
                                 consumer_timeout_ms=1000 * self.args.execution_time,
                                 bootstrap_servers=[self.args.bootstrap_servers],
                                 fetch_max_wait_ms=self.args.fetch_wait_max_ms,
                                 max_partition_fetch_bytes=self.args.max_partition_fetch_bytes,
                                 max_poll_records=self.args.max_poll_records)

        os.system('rm *.log')
        start = time.time()
        # move offset to the end
        consumer.poll()
        consumer.seek_to_end()

        predict_thread = Thread(target=self.bg_predict, daemon=True, name='image prediction')
        predict_thread.start()
        while time.time() - start < self.args.execution_time:
            message_batch = consumer.poll()
            poll_time = time.time()
            for i, partition_batch in enumerate(message_batch.values()):
                for message in partition_batch:
                    raw_msg = pickle.loads(message.value)
                    if not self.last_timestamp or (time.time() - self.last_timestamp > INTERVAL):
                        self.capture_metrics(consumer.metrics(), poll_time, message.timestamp)
                    self.img_queue.append(raw_msg['img'])
        consumer.close()

    def bg_predict(self):
        while True:
            try:
                if len(self.img_queue) > 0:
                    img = self.img_queue.popleft()
                    predict(self.model, img_matrix=img)
            except Exception as ex:
                print(ex)

    def get_latency(self):
        rec_size = len(self.latency_samples)
        return utils.process_metrics(self.latency_samples[int(0.1*rec_size):int(0.9*rec_size)])

    def get_throughput(self):
        rec_size = len(self.throughput_samples)
        return utils.process_metrics(self.throughput_samples[int(0.1*rec_size):int(0.9*rec_size)])

    def get_e2e_latency(self):
        rec_size = len(self.e2e_latency_samples)
        return utils.process_metrics(self.e2e_latency_samples[int(0.1*rec_size):int(0.9*rec_size)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_id', type=str, required=True)
    parser.add_argument('--bootstrap_servers', type=str, default='localhost:9092')
    parser.add_argument('--topic', type=str, default='distracted_driver_detection')
    parser.add_argument('--execution_time', type=int, default=120)

    parser.add_argument('--fetch_wait_max_ms', type=int, default=500)
    parser.add_argument('--max_partition_fetch_bytes', type=int, default=1048576)
    parser.add_argument('--max_poll_records', type=int, default=500)
    args = parser.parse_args()

    sub = MyConsumer(args)
    sub.consume_msg()

    latency = []
    throughput = []
    e2e_latency = []

    latency.append(sub.get_latency())
    throughput.append(sub.get_throughput())
    e2e_latency.append(sub.get_e2e_latency())

    latency = np.array(latency).mean(axis=0).reshape(1, -1)
    throughput = np.array(throughput).mean(axis=0).reshape(1, -1)
    e2e_latency = np.array(e2e_latency).mean(axis=0).reshape(1, -1)

    np.savetxt('latency.log', latency, fmt='%.3f', delimiter=',')
    np.savetxt('throughput.log', throughput, fmt='%.3f', delimiter=',')
    np.savetxt('e2e_latency.log', e2e_latency, fmt='%.3f', delimiter=',')