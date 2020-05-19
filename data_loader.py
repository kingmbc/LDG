import numpy as np
import datetime
import torch
import torch.utils
from datetime import datetime, timezone


class EventsDataset(torch.utils.data.Dataset):
    '''
    Base class for event datasets
    '''
    def __init__(self, TZ=None):
        self.TZ = TZ  # timezone.utc

        # Implement here these fields (see examples in actual datasets):
        # self.FIRST_DATE = datetime()
        # self.TEST_TIMESLOTS = []
        # self.N_nodes = 100
        # self.A_initial = np.random.randint(0, 2, size=(self.N_nodes, self.N_nodes))
        # self.A_last = np.random.randint(0, 2, size=(self.N_nodes, self.N_nodes))
        #
        # self.all_events = []
        # self.n_events = len(self.all_events)
        #
        # self.event_types = ['communication event']
        # self.event_types_num = {'association event': 0}
        # k = 1  # k >= 1 for communication events
        # for t in self.event_types:
        #     self.event_types_num[t] = k
        #     k += 1


    def get_Adjacency(self, multirelations=False):
        return None, None, None

    def __len__(self):
        return self.n_events

    def __getitem__(self, index):
        """
        :param index:
        :사용하는 데이터 및 그에 따른 아웃풋
        :Input: self.all_events(전체 이벤트),
                self.time_bar (모든 노드 이전 시간),
                self_FIRST_DATE (시작일),
                self.event_types_num(데이터의 이벤트 타입 인덱스)
        :return: u, v (해당 이벤트 노드)
                time_delta_uv (u, v 자신의 이전 이벤트 시간과의 차이),
                k (Relation),
                time_bar (모든 노드 이전시간),
                time_cur (현재 이벤트 시간)
        """
        tpl = self.all_events[index]
        u, v, rel, time_cur = tpl

        # Compute time delta in seconds (t_p - \bar{t}_p_j) that will be fed to W_t
        # u와 v노드 각각 자신의 바로 이전 이벤트와 현재 이벤트 시간과의 차이를 저장
        time_delta_uv = np.zeros((2, 4))  # two nodes x 4 values (일, 시, 분, 초)

        # most recent previous time for all nodes
        # main.py에서 초기화됨, time_bar = np.zeros((dataset.N_nodes, 1)) + dataset.FIRST_DATE.timestamp()
        time_bar = self.time_bar.copy()
        assert u != v, (tpl, rel)       #u와 v는 같지 않아야 한다! 만약 같다면 에러처!

        # u와 v노드, 각각에 대해 time_delta_uv와 time_bar를 계산하면서 저장함
        for c, j in enumerate([u, v]):  #노드 u와 v, 각각에 대해
            t = datetime.fromtimestamp(self.time_bar[j], tz=self.TZ)
            if t.toordinal() >= self.FIRST_DATE.toordinal():  # assume no events before FIRST_DATE
                td = time_cur - t       #데이터.현재 시간 - 데이터.이전 시간 (toordinal()는 1년1월1일부터 누적날짜를 리턴. 그래서 1은 하루를 의미)
                time_delta_uv[c] = np.array([td.days,  # total number of days, still can be a big number
                                             td.seconds // 3600,  # hours, max 24
                                             (td.seconds // 60) % 60,  # minutes, max 60
                                             td.seconds % 60],  # seconds, max 60
                                            np.float)
                # assert time_delta_uv.min() >= 0, (index, tpl, time_delta_uv[c], node_global_time[j])
            else:
                raise ValueError('unexpected result', t, self.FIRST_DATE)
            self.time_bar[j] = time_cur.timestamp()  # last time stamp for nodes u and v

        k = self.event_types_num[rel]

        # sanity checks
        assert np.float64(time_cur.timestamp()) == time_cur.timestamp(), (
            np.float64(time_cur.timestamp()), time_cur.timestamp())

        time_cur = np.float64(time_cur.timestamp())
        time_bar = time_bar.astype(np.float64)
        time_cur = torch.from_numpy(np.array([time_cur])).double()

        assert time_bar.max() <= time_cur, (time_bar.max(), time_cur)
        return u, v, time_delta_uv, k, time_bar, time_cur
