from Model import Model
from ModelA import ModelA
from ModelB import ModelB
import data_parser as prs
if __name__ == '__main__':

    print("First Model:")
    modelA = ModelA('modelA')
    modelA.test('train.wtag', 'test.wtag')
    modelA.comp('comp.words')

    print("Second Model:")
    modelB = ModelB('modelB')
    modelB.test('train2.wtag')
    modelB.clear()
    train_data = prs.parse('train2.wtag')
    modelB.train(train_data)
    modelB.comp('comp2.words')

