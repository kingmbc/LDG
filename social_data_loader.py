import matplotlib.pyplot as plt
import os
from os.path import join as pjoin
import numpy as np
import datetime
import pickle
import pandas
import itertools
import torch
import torch.utils
from data_loader import EventsDataset


class SocialEvolutionDataset(EventsDataset):
    '''
    Class to load batches for training and testing
    '''

    FIRST_DATE = datetime.datetime(2008, 9, 11)  # consider events starting from this time
    EVENT_TYPES = ['SMS', 'Proximity', 'Calls']

    def __init__(self,
                 subj_features,
                 data: dict,
                 MainAssociation: str,
                 data_train=None):  #현재 데이터가 test_data인경우 data_train에는 None이 아니고, data_train값이 들어감
        """

        :param subj_features: Node의 최초 임베딩값 (이 데이터의 경우에는 사용자 Profile (Subject.csv)에서 가져온다)
        :param data: 아래 static method인 load_data()로 부터 읽어들인 데이터 (Train_data일수도 있고, Test_data일수도 있다.)
                    특히, data안에는 'initial_embeddings', 'train', 'test'가 Key로 들어가있고,
                    'train'/'test' 각각 안에는 Adj(시간별/관계별), EVENT_TYPES, relations가 들어가 있음
        :param MainAssociation: Association으로 간주할 Relation 명 (예, 'CloseFriend')
        :param data_train: None이 아니라면, 현재 주어진 데이터는 Test_data라는 것이고, 이 data_train을 이용해 마지막 이벤트의 날짜 및 Adj.를 가져온다.
        """
        super(SocialEvolutionDataset, self).__init__()

        self.subj_features = subj_features
        self.data = data
        self.all_events = []
        self.event_types_num = {}
        self.time_bar = None
        self.MainAssociation = MainAssociation
        self.TEST_TIMESLOTS = [datetime.datetime(2009, 5, 10),
                               datetime.datetime(2009, 5, 20),
                               datetime.datetime(2009, 5, 31),
                               datetime.datetime(2009, 6, 10),
                               datetime.datetime(2009, 6, 20),
                               datetime.datetime(2009, 6, 30)]
        self.FIRST_DATE = SocialEvolutionDataset.FIRST_DATE
        self.event_types = SocialEvolutionDataset.EVENT_TYPES

        """ FIRST_DATE 이후의 이벤트만 가지도록 필터링 한 후, event_types_num에 Dictionary로 저"""
        k = 1  # k >= 1 for communication events
        for t in self.event_types:
            print(t, k, len(data.EVENT_TYPES[t].tuples))

            #FIRST_DATE보다 큰 이벤트만 저장한다.
            events = list(filter(lambda x: x[3].toordinal() >= self.FIRST_DATE.toordinal(),
                                 data.EVENT_TYPES[t].tuples))
            self.all_events.extend(events)
            self.event_types_num[t] = k
            k += 1

        n = len(self.all_events)
        print(set([e[2] for e in self.all_events]))
        assert MainAssociation not in (set([e[2] for e in self.all_events])) #이때에는 Association Event가 없다.
        self.N_nodes = subj_features.shape[0]

        if data.split == 'train':
            Adj_all, keys, Adj_all_last = self.get_Adjacency()

            print('initial and final associations', self.MainAssociation,
                  Adj_all.sum(), Adj_all_last.sum(),
                  np.allclose(Adj_all, Adj_all_last))


        # Initial topology
        if len(list(data.Adj.keys())) > 0:
            #list(data.Adj.keys())[0] == 첫번째 설문 날짜, 이를 data.Adj에서 가져와서, 그것의 Key들을 가져오면, 그때 응답된 Relation들을 가져옴
            keys = sorted(list(data.Adj[list(data.Adj.keys())[0]].keys()))  # relation keys
            keys.remove(MainAssociation)
            keys = [MainAssociation] + keys  # to make sure CloseFriend goes first

            k = 0  # k <= 0 for association events
            for rel in keys:
                if rel != MainAssociation:
                    continue
                if data_train is None:
                    date = sorted(list(data.Adj.keys()))[0]  # first date
                    Adj_prev = data.Adj[date][rel]
                else:
                    date = sorted(list(data_train.Adj.keys()))[-1]  # last date of the training set
                    Adj_prev = data_train.Adj[date][rel]
                self.event_types_num[rel] = k   #Association Relation을 이때 처음 저장

                N = Adj_prev.shape[0]

                # Associative events
                #TODO: RelationshipFromSurveys.csv에서 저장한 시간별 친구 관계 정보 (data.Adj)를 바탕으로,
                # 시간이 흘러가면서 언제 친구를 맺었는지 이벤트 정보를 기록함
                # 한계점: 설문데이터이다 보니, 친구 맺은 이벤트 시간이 정확하지는 않음
                for date_id, date in enumerate(sorted(list(data.Adj.keys()))):  # start from the second survey
                    if date.toordinal() >= self.FIRST_DATE.toordinal():
                        # for rel_id, rel in enumerate(sorted(list(dygraphs.Adj[date].keys()))):
                        assert data.Adj[date][rel].shape[0] == N
                        for u in range(N):
                            for v in range(u + 1, N):
                                # if two nodes become friends, add the event
                                if data.Adj[date][rel][u, v] > 0 and Adj_prev[u, v] == 0:
                                    assert u != v, (u, v, k)
                                    self.all_events.append((u, v, rel, date))

                    Adj_prev = data.Adj[date][rel]

                k -= 1

                #원래 처음 communicative event만 있었던 n에서, associative event가 추가된 all_events갯수를 뺌
                print(data.split, rel, len(self.all_events) - n)
                n = len(self.all_events)

        self.all_events = sorted(self.all_events, key=lambda x: int(x[3].timestamp()))

        print(set([e[2] for e in self.all_events]))
        assert MainAssociation in (set([e[2] for e in self.all_events])) #이때에는 Association Event가 있다.
        print('%d events' % len(self.all_events))
        print('last 10 events:')
        for event in self.all_events[-10:]:
            print(event)

        self.n_events = len(self.all_events)

        H_train = np.zeros((N, N))
        evt_count = 0
        for e in self.all_events:
            H_train[e[0], e[1]] += 1
            H_train[e[1], e[0]] += 1
            evt_count += 1
        print(f'H_train {evt_count}, {H_train.max()}, {H_train.min()}, {H_train.std()}')
        self.H_train = H_train  #각 사용자 간의 상호작용 횟수를 모두 더함 (이벤트 종류 상관없이!!)


    @staticmethod
    def load_data(data_dir, prob, dump=True):
        """
        데이터를 이미 저장된 pkl에서 불러오거나, 없다면 새로이 만드는 함수.
        :param data_dir: 위치 경로
        :param prob: 데이터에서 이벤트의 생성 Confidence 임계값 (센서 이벤트가 많기 때문에 신뢰성 높은 이벤트를 고려하기 위함)
        :param dump: 파일을 저장할지 여부
        :return: data(
        """
        data_file = pjoin(data_dir, 'data_prob%s.pkl' % prob)
        if os.path.isfile(data_file):
            print('loading data from %s' % data_file)
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
        else:
            data = {'initial_embeddings': SubjectsReader(pjoin(data_dir, 'Subjects.csv')).features_onehot}
            for split in ['train', 'test']:
                data.update(
                    {split: SocialEvolution(data_dir, split=split, MIN_EVENT_PROB=prob)})
            if dump:
                # dump data files to avoid their generation again
                print('saving data to %s' % data_file)
                with open(data_file, 'wb') as f:
                    pickle.dump(data, f, protocol=2)  # for compatibility
        return data

    def get_Adjacency(self, multirelations=False):
        """
        :param multirelations: Association을 고려할때 여러개의 relation 으로 할 것인지?
        :return:
            Adj_all_init: 첫 날짜의 Adjacency
            keys: Relations들
            Adj_all_last: 마지막 날짜의 Adjacency
        """
        dates = sorted(list(self.data.Adj.keys()))
        Adj_all_init = self.data.Adj[dates[0]]           # 첫 날짜의 Adjacency
        Adj_all_last = self.data.Adj[dates[-1]]     # 마지막 설문 날짜의 Adjacency
        # Adj_friends = Adj_all_init[self.MainAssociation].copy()
        if multirelations:
            keys = sorted(list(Adj_all_init.keys()))
            keys.remove(self.MainAssociation)
            keys = [self.MainAssociation] + keys  # to make sure CloseFriend goes first
            Adj_all_init = np.stack([Adj_all_init[rel].copy() for rel in keys], axis=2)
            Adj_all_last = np.stack([Adj_all_last[rel].copy() for rel in keys], axis=2)
        else:
            keys = [self.MainAssociation]
            Adj_all_init = Adj_all_init[self.MainAssociation].copy()
            Adj_all_last = Adj_all_last[self.MainAssociation].copy()

        return Adj_all_init, keys, Adj_all_last


    def time_to_onehot(self, d):
        x = []
        for t, max_t in [(d.weekday(), 7), (d.hour, 24), (d.minute, 60), (d.second, 60)]:
            x_t = np.zeros(max_t)
            x_t[t] = 1
            x.append(x_t)
        return np.concatenate(x)


class CSVReader:
    '''
    General class to read any relationship csv in this dataset
    '''

    def __init__(self,
                 csv_path,
                 split,  # 'train', 'test', 'all'
                 MIN_EVENT_PROB,
                 event_type=None,
                 N_subjects=None,
                 test_slot=1):
        self.csv_path = csv_path
        print(os.path.basename(csv_path))

        if split == 'train':
            time_start = 0
            time_end = datetime.datetime(2009, 4, 30).toordinal()
        elif split == 'test':
            if test_slot != 1:
                raise NotImplementedError('test on time slot 1 for now')
            time_start = datetime.datetime(2009, 5, 1).toordinal()
            time_end = datetime.datetime(2009, 6, 30).toordinal()
        else:
            time_start = 0
            time_end = np.Inf

        csv = pandas.read_csv(csv_path)
        self.data = {}
        to_date1 = lambda s: datetime.datetime.strptime(s, '%Y-%m-%d')
        to_date2 = lambda s: datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
        user_columns = list(filter(lambda c: c.find('user') >= 0 or c.find('id') >= 0, list(csv.keys())))
        assert len(user_columns) == 2, (list(csv.keys()), user_columns)
        self.time_column = list(filter(lambda c: c.find('time') >= 0 or c.find('date') >= 0, list(csv.keys())))
        assert len(self.time_column) == 1, (list(csv.keys()), self.time_column)
        self.time_column = self.time_column[0]

        self.prob_column = list(filter(lambda c: c.find('prob') >= 0, list(csv.keys())))

        for column in list(csv.keys()):
            values = csv[column].tolist()
            for fn in [int, float, to_date1, to_date2]:
                try:
                    values = list(map(fn, values))
                    break
                except Exception as e:
                    continue
            self.data[column] = values

        n_rows = len(self.data[self.time_column])

        time_stamp_days = np.array([d.toordinal() for d in self.data[self.time_column]], dtype=np.int)

        # skip data where one of users is missing (nan) or interacting with itself or timestamp not in range
        conditions = [~np.isnan(self.data[user_columns[0]]),
                      ~np.isnan(self.data[user_columns[1]]),
                      np.array(self.data[user_columns[0]]) != np.array(self.data[user_columns[1]]),
                      time_stamp_days >= time_start,
                      time_stamp_days <= time_end]

        if len(self.prob_column) == 1:
            print(split, event_type, self.prob_column)
            # skip data if the probability of event is 0 or nan (available for some event types)
            conditions.append(np.nan_to_num(np.array(self.data[self.prob_column[0]])) > MIN_EVENT_PROB)

        valid_ids = np.ones(n_rows, dtype=np.bool)
        for cond in conditions:
            valid_ids = valid_ids & cond

        self.valid_ids = np.where(valid_ids)[0]

        time_stamps_sec = [self.data[self.time_column][i].timestamp() for i in self.valid_ids]
        self.valid_ids = self.valid_ids[np.argsort(time_stamps_sec)]

        print(split, len(self.valid_ids), n_rows)

        for column in list(csv.keys()):
            values = csv[column].tolist()
            key = column + '_unique'
            for fn in [int, float, to_date1, to_date2]:
                try:
                    values = list(map(fn, values))
                    break
                except Exception as e:
                    continue

            self.data[column] = values

            values_valid = [values[i] for i in self.valid_ids]
            self.data[key] = np.unique(values_valid)
            print(key, type(values[0]), len(self.data[key]), self.data[key])

        self.subjects, self.time_stamps = [], []
        for usr_col in range(len(user_columns)):
            self.subjects.extend([self.data[user_columns[usr_col]][i] for i in self.valid_ids])
            self.time_stamps.extend([self.data[self.time_column][i] for i in self.valid_ids])

        # set O={(u, v, k, t)}
        self.tuples = []
        if N_subjects is not None:
            # Compute frequency of communcation between users
            print('user_columns', user_columns)
            self.Adj = np.zeros((N_subjects, N_subjects))
            for row in self.valid_ids:
                subj1 = self.data[user_columns[0]][row]
                subj2 = self.data[user_columns[1]][row]

                assert subj1 != subj2, (subj1, subj2)
                assert subj1 > 0 and subj2 > 0, (subj1, subj2)
                try:
                    self.Adj[int(subj1) - 1, int(subj2) - 1] += 1
                    self.Adj[int(subj2) - 1, int(subj1) - 1] += 1
                except:
                    print(subj1, subj2)
                    raise

                self.tuples.append((int(subj1) - 1,
                                    int(subj2) - 1,
                                    event_type,
                                    self.data[self.time_column][row]))

        n1 = len(self.tuples)
        self.tuples = list(set(itertools.chain(self.tuples)))
        self.tuples = sorted(self.tuples, key=lambda t: t[3].timestamp())
        n2 = len(self.tuples)
        print('%d/%d duplicates removed' % (n1 - n2, n1))


class SubjectsReader:
    '''
    Class to read Subjects.csv in this dataset
    이 CSV파일에서 'user'를 포함하는 컬럼에 대해 one-hot encoding해서 가져오게
    '''

    def __init__(self,
                 csv_path):
        self.csv_path = csv_path
        print(os.path.basename(csv_path))

        csv = pandas.read_csv(csv_path)

        #subjects = csv['user_id'].tolist() 와 동일함
        subjects = csv[list(filter(lambda column: column.find('user') >= 0, list(csv.keys())))[0]].tolist()
        print('Number of subjects', len(subjects))
        features = []
        for column in list(csv.keys()):
            if column.find('user') >= 0:    #'user'를 포함하는 컬럼만 하겠다는 의미
                continue
            values = list(map(str, csv[column].tolist()))   #user_id의 숫자를 문자로 변경
            features_unique = np.unique(values)
            features_onehot = np.zeros((len(subjects), len(features_unique)))   #
            for subj, feat in enumerate(values):
                ind = np.where(features_unique == feat)[0]
                assert len(ind) == 1, (ind, features_unique, feat, type(feat))
                features_onehot[subj, ind[0]] = 1
            features.append(features_onehot)       #결국, 여기에는 'user'가 포함된 컬럼을 one_hot_eocoding시키고, features에 인코딩된 컬럼하나씩 쌓는다.

        features_onehot = np.concatenate(features, axis=1)
        print('features', features_onehot.shape)
        self.features_onehot = features_onehot


class SocialEvolution():
    '''
    Class to read all csv in this dataset
    모든 CSV파일을 파싱하여 데이터로 저장한다.
    RelationshipsFromSurveys.csv = Adjacency를 계산하기 위해 사용되고,
    나머지 csv파일들은 이벤트 스트림을 저장히기 위해 사용됨
    '''

    def __init__(self,
                 data_dir:str,
                 split:str,
                 MIN_EVENT_PROB:int):
        self.data_dir = data_dir
        self.split = split
        self.MIN_EVENT_PROB = MIN_EVENT_PROB

        self.relations = CSVReader(pjoin(data_dir, 'RelationshipsFromSurveys.csv'), split=split, MIN_EVENT_PROB=MIN_EVENT_PROB)
        self.relations.subject_ids = np.unique(self.relations.data['id.A'] + self.relations.data['id.B'])   #서로간의 관계 데이터에서 Unique 사용자들
        self.N_subjects = len(self.relations.subject_ids)
        print('Number of subjects', self.N_subjects)

        self.__get_communicative_events(data_dir, split, MIN_EVENT_PROB)
        self.__get_adjacency_matrix()


    def __get_adjacency_matrix(self):
        # Compute adjacency matrices for associative relationship data
        self.Adj = {}
        dates = self.relations.data['survey.date']  #각 설문응답별 설문날짜들
        rels = self.relations.data['relationship']  #각 설문응답에서의 관계응답들(e.g., BlogLivejournalTwitter,FacebookAllTaggedPhotos)
        for date_id, date in enumerate(self.relations.data['survey.date_unique']):
            self.Adj[date] = {}
            ind = np.where(np.array([d == date for d in dates]))[0] #위 unique날짜에 해당하는 응답들
            for rel_id, rel in enumerate(self.relations.data['relationship_unique']):
                ind_rel = np.where(np.array([r == rel for r in [rels[i] for i in ind]]))[0] #응답들 중에서 특정 관계(rel)에 대응되는 응답들
                A = np.zeros((self.N_subjects, self.N_subjects))
                for j in ind_rel:   #각 관계응답(예를 들어, BlogLivejournalTwitter) 인덱스 번호를 가져와서,
                    row = ind[j]    #결국 이 row는 특정 날짜/특정 관계에 찾아진 응답 인덱스임. 그래서 아래에 그때 연관된 두사람 간의 연결을 맺어줌
                    A[self.relations.data['id.A'][row] - 1, self.relations.data['id.B'][row] - 1] = 1
                    A[self.relations.data['id.B'][row] - 1, self.relations.data['id.A'][row] - 1] = 1
                self.Adj[date][rel] = A     #특정 날짜(date)와 특정 관계(rel)에 대한 Adjacency(A)를 저장
                # sanity check
                for row in range(len(dates)):
                    if rels[row] == rel and dates[row] == date:
                        #즉, 아래는 Adj[date][rel][u1, u2] == 1 이어야 함(위에서 그렇게 저장했으므로!)
                        assert self.Adj[dates[row]][rels[row]][
                                   self.relations.data['id.A'][row] - 1, self.relations.data['id.B'][row] - 1] == 1
                        assert self.Adj[dates[row]][rels[row]][
                                   self.relations.data['id.B'][row] - 1, self.relations.data['id.A'][row] - 1] == 1

    def __get_communicative_events(self, data_dir, split, MIN_EVENT_PROB):
        # Read communicative events
        self.EVENT_TYPES = {}
        for t in SocialEvolutionDataset.EVENT_TYPES:
            self.EVENT_TYPES[t] = CSVReader(pjoin(data_dir, '%s.csv' % t),
                                           split=split,
                                           MIN_EVENT_PROB=MIN_EVENT_PROB,
                                           event_type=t,
                                           N_subjects=self.N_subjects)