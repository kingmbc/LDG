import os
import numpy as np
import datetime
import pickle
from datetime import datetime, timezone
import dateutil.parser
from data_loader import EventsDataset


def iso_parse(dt):
    # return datetime.fromisoformat(dt)  # python >= 3.7
    return dateutil.parser.isoparse(dt)

class GithubDataset(EventsDataset):

    def __init__(self, split, data_dir='./Github'):
        super(GithubDataset, self).__init__()

        if split == 'train':
            time_start = 0
            time_end = datetime(2013, 8, 31, tzinfo=self.TZ).toordinal()
        elif split == 'test':
            time_start = datetime(2013, 9, 1, tzinfo=self.TZ).toordinal()
            time_end = datetime(2014, 1, 1, tzinfo=self.TZ).toordinal()
        else:
            raise ValueError('invalid split', split)

        self.FIRST_DATE = datetime(2012, 12, 28, tzinfo=self.TZ)

        self.TEST_TIMESLOTS = [datetime(2013, 9, 1, tzinfo=self.TZ),
                               datetime(2013, 9, 25, tzinfo=self.TZ),
                               datetime(2013, 10, 20, tzinfo=self.TZ),
                               datetime(2013, 11, 15, tzinfo=self.TZ),
                               datetime(2013, 12, 10, tzinfo=self.TZ),
                               datetime(2014, 1, 1, tzinfo=self.TZ)]


        """
            users_events는 dictionary로서, users_events['jkroso']로 값을 가져온다
            그리고 그 내부는 list이라서, users_events['jkroso'][0] 으로 인덱싱해서 가져온다.
            
            event_types는 dictionary, event_types['PushEvent']
            users_follow도 dictionary, users_follow['lgs'][0]으로 인덱싱
        """
        #
        with open(os.path.join(data_dir, 'github_284users_events_2013.pkl'), 'rb') as f:
            users_events, event_types = pickle.load(f)

        with open(os.path.join(data_dir, 'github_284users_follow_2011_2012.pkl'), 'rb') as f:
            users_follow = pickle.load(f)

        print(event_types)

        #데이터 내부에는 event가 숫자로 되어있어서 매핑 테이블 필요
        self.events2name = {}
        #아래 코드와 동일 -> events2name = {y:x for x, y in event_type.items()}
        for e in event_types:
            self.events2name[event_types[e]] = e
        print(self.events2name)

        self.event_types = ['ForkEvent', 'PushEvent', 'WatchEvent', 'IssuesEvent', 'IssueCommentEvent',
                           'PullRequestEvent', 'CommitCommentEvent']
        self.assoc_types = ['FollowEvent']
        #d['type']에는 이벤트의 인덱스 번호가 들어감
        self.is_comm = lambda d: self.events2name[d['type']] in self.event_types
        self.is_assoc = lambda d: self.events2name[d['type']] in self.assoc_types

        #사용자 ID와 인덱스 번호의 매핑이 dictionary로 저장됨
        user_ids = {}
        for id, user in enumerate(sorted(users_events)):
            user_ids[user] = id

        self.N_nodes = len(user_ids)

        #TODO: Adjacency_Initial과 Adjancency_Last를 채워 넣는다.
        """A_initial에는 Association('FollowEvent')에 대한 기록을 넣는다""
        ""예, u1이 u2를 follow하고 있으면 1, 그렇지 않으면 0"""
        self.A_initial = np.zeros((self.N_nodes, self.N_nodes))
        for user in users_follow:
            for e in users_follow[user]:    #e는 {'created_at': '2012-12-09T07:17:41-08:00', 'type': 'FollowEvent', 'login': 'n33'} 처럼 생겼음
                assert e['type'] in self.assoc_types, e['type']
                #사용자가 followe한 상대방 유저가 전체 사용자 이벤트 중에 있다면 A_initial에 1로 채움
                if e['login'] in users_events:
                    self.A_initial[user_ids[user], user_ids[e['login']]] = 1

        """A_last에는 Communication에 대한 기록을 통해 값을 채운다."""
        """예, u1이 u2를 follow했으면 1, 그렇지 않으면 0"""
        self.A_last = np.zeros((self.N_nodes, self.N_nodes))
        for user in users_events:
            for e in users_events[user]:
                if self.events2name[e['type']] in self.assoc_types:
                    self.A_last[user_ids[user], user_ids[e['login']]] = 1

        print('\nA_initial', np.sum(self.A_initial))
        print('A_last', np.sum(self.A_last), '\n')

        #TODO: users_events에 있는 각 사용자별 이벤트를 돌면서, (u1, u2, EventType, Time)을 채워넣음
        all_events = []
        for user in users_events:
            if user not in user_ids:    #위에서 모든 user_ids를 user_events로부터 넣었기 때문에 이 케이스는 사실상 무시됨
                continue
            user_id = user_ids[user]
            for ind, event in enumerate(users_events[user]):
                event['created_at'] = datetime.fromtimestamp(event['created_at']) #timestamp->datetime으로 변경
                if event['created_at'].toordinal() >= time_start and event['created_at'].toordinal() <= time_end:
                    if 'owner' in event:
                        if event['owner'] not in user_ids:
                            continue
                        user_id2 = user_ids[event['owner']]     #'owner'는 상대방 ID을 의미함
                    elif 'login' in event:
                        if event['login'] not in user_ids:
                            continue
                        user_id2 = user_ids[event['login']]     #'login'는 상대방 ID을 의미함
                    else:
                        raise ValueError('invalid event', event)
                    if user_id != user_id2:
                        all_events.append((user_id,
                                           user_id2,
                                           self.events2name[event['type']],
                                           event['created_at']))

        self.all_events = sorted(all_events, key=lambda t: t[3].timestamp())
        print('\n%s' % split.upper())
        print('%d events between %d users loaded' % (len(self.all_events), self.N_nodes))
        print('%d communication events' % (len([t for t in self.all_events if t[2] == 'FollowEvent'])))
        print('%d assocition events' % (len([t for t in self.all_events if t[2] != 'FollowEvent'])))

        self.event_types_num = {self.assoc_types[0]: 0}
        k = 1  # k >= 1 for communication events
        for t in self.event_types:
            self.event_types_num[t] = k
            k += 1

        self.n_events = len(self.all_events)


    def get_Adjacency(self, multirelations=False):
        if multirelations:
            print('warning: Github has only one relation type (FollowEvent), so multirelations are ignored')
        return self.A_initial, self.assoc_types, self.A_last
