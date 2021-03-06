import numpy as np
import datetime
from datetime import datetime, timezone
from data_loader import EventsDataset


class ExampleDataset(EventsDataset):

    def __init__(self, split, data_dir=None):
        super(ExampleDataset, self).__init__()

        if split == 'train':
            time_start = 0
            # TODO: Training Data의 종료일 기입
            time_end = datetime(2013, 8, 31, tzinfo=self.TZ).toordinal()
        elif split == 'test':
            # TODO: Test 데이터의 시작일/종료일 기입
            time_start = datetime(2013, 9, 1, tzinfo=self.TZ).toordinal()
            time_end = datetime(2014, 1, 1, tzinfo=self.TZ).toordinal()
        else:
            raise ValueError('invalid split', split)

        #TODO: 데이터의 시작일 기입
        self.FIRST_DATE = datetime(2012, 12, 28, tzinfo=self.TZ)

        #TODO: 시간별로 테스트데이터를 n개의 Slot으로 나눔 (n=6)
        self.TEST_TIMESLOTS = [datetime(2013, 9, 1, tzinfo=self.TZ),
                               datetime(2013, 9, 25, tzinfo=self.TZ),
                               datetime(2013, 10, 20, tzinfo=self.TZ),
                               datetime(2013, 11, 15, tzinfo=self.TZ),
                               datetime(2013, 12, 10, tzinfo=self.TZ),
                               datetime(2014, 1, 1, tzinfo=self.TZ)]

        self.N_nodes = 100

        self.A_initial = np.random.randint(0, 2, size=(self.N_nodes, self.N_nodes))
        self.A_last = np.random.randint(0, 2, size=(self.N_nodes, self.N_nodes))

        print('\nA_initial', np.sum(self.A_initial))
        print('A_last', np.sum(self.A_last), '\n')

        #TODO: 이벤트 갯수 정함
        self.n_events = 10000
        all_events = []
        for i in range(self.n_events):
            user_id1 = np.random.randint(0, self.N_nodes)
            user_id2 = np.random.choice(np.delete(np.arange(self.N_nodes), user_id1))   #u1이 아닌 사람들 중에 한명
            ts = max((time_start, self.FIRST_DATE.toordinal())) #시작시간을 정함
            # 시작시간 ~ 끝시간 사이값으로 이벤트 정함
            event_time = datetime.fromordinal(ts + np.random.randint(0, time_end - ts))
            #이벤트 시간이 시작일보다는 커야 한다!
            assert event_time.timestamp() >= self.FIRST_DATE.timestamp(), (event_time, self.FIRST_DATE)
            all_events.append((user_id1, user_id2, np.random.choice(['communication event',
                                                                     'association event']), event_time))

        #TODO: 모든 이벤트 타입에 대해 여기다가 추가함!
        self.event_types = ['communication event']

        self.all_events = sorted(all_events, key=lambda t: t[3].timestamp())
        print('\n%s' % split.upper())
        print('%d events between %d users loaded' % (len(self.all_events), self.N_nodes))
        print('%d communication events' % (len([t for t in self.all_events if t[2] == 1])))
        print('%d assocition events' % (len([t for t in self.all_events if t[2] == 0])))

        #'association event'는 0으로 지정해두고, 나머지 이벤트는 1부터 증가하는 값으로 인덱싱
        self.event_types_num = {'association event': 0}
        k = 1  # k >= 1 for communication events
        for t in self.event_types:
            self.event_types_num[t] = k
            k += 1

        self.n_events = len(self.all_events)

    def get_Adjacency(self, multirelations=False):
        if multirelations:
            print('warning: this dataset has only one relation type, so multirelations are ignored')
        return self.A_initial, ['association event'], self.A_last
