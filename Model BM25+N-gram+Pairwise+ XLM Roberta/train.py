'''
author: @PenguinsResearch © 2022
'''

import dill
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import HashingVectorizer

data = pd.read_csv('/code/zac2022_context_question_category.csv')
p_test = 0.2
trainset = data.sample(frac=1-p_test, random_state=42)
testset = data.drop(trainset.index)

train_x, train_y = [], []
test_x, test_y = [], []

for index, row in trainset.iterrows():
    train_x.append(row['question'])
    train_y.append(row['category'])

for index, row in testset.iterrows():
    test_x.append(row['question'])
    test_y.append(row['category'])

Encoder = LabelEncoder()

train_y = Encoder.fit_transform(train_y)
test_y = Encoder.fit_transform(test_y)

hashing_vect = HashingVectorizer()
hashing_vect.fit(train_x)

Train_X_Hashing = hashing_vect.transform(train_x)
Test_X_Hashing = hashing_vect.transform(test_x)

SVM = svm.SVC(C=2.0, kernel='linear', degree=10, gamma='auto', verbose=True)
SVM.fit(Train_X_Hashing, train_y)
predictions_SVM = SVM.predict(Test_X_Hashing)
print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, test_y)*100)


class Classifier:
    def __init__(self):
        self.encoder = Encoder
        self.tfidf = hashing_vect
        self.svm = SVM

    def hand_check(self, questions: list):
        ans = []
        for qe in questions:
            if self.is_datetime(qe):
                ans.append('datetime')
            elif self.is_query_quantity(qe):
                ans.append('quantity')
            else:
                ans.append(None)

    def is_datetime(self, question):
        _question = [
            'khi nào',
            'lúc nào',
            'bao giờ',
            'thời gian nào',
            'ngày mấy',
            'ngày nào',
            'ngày bao nhiêu',
            'ngày tháng năm nào',
            'tháng nào',
            'tháng bao nhiêu',
            'năm nào',
            'năm bao nhiêu',
        ]
        for word in _question:
            if word in question:
                return True
        return False

    def is_query_quantity(self, question):
        _question = [
            'bao nhiêu',
            'bao lâu',
        ]

        for word in _question:
            if word in question:
                return True

    def is_wiki(self, question):
        _question = [
            'vào năm',
        ]
        for word in _question:
            if word in question:
                return True

    def predict(self, questions: list):
        if isinstance(questions, str):
            questions = [questions]
        ans = []
        for qe in questions:
            if self.is_datetime(qe):
                ans.append('datetime')
            elif self.is_query_quantity(qe):
                ans.append('quantity')
            elif self.is_wiki(qe):
                ans.append('wiki')
            else:
                ans.append(self.encoder.inverse_transform(
                    self.svm.predict(self.tfidf.transform([qe])))[0])
        return ans

    def __call__(self, question):
        return self.predict(question)


# save the model to disk
dill.dump(Classifier(), open(
    '/code/saved_models/classifier/classifier_model.sav', 'wb'))

# test
test_question = 'Tây Du Ký phiên bản Trương Kỷ Trung khởi quay năm nào'
print(Classifier()(test_question))

print("accuracy: ", accuracy_score(Classifier()(
    testset['question']), testset['category']))
