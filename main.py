from Model import Model
from ModelA import ModelA
from ModelB import ModelB
import data_parser as prs
if __name__ == '__main__':
    with open('res_fileA','w') as f:
        print("First Model:", file=f)
        modelA = ModelA('modelA', True)
        modelA.test('train.wtag', 'test.wtag',f)
        modelA.comp('comp.words')
    with open('res_fileB', 'w') as f:
        print("Second Model:", file=f)
        modelB = ModelB('modelB')
        modelB.test('train2.wtag', None,f)
        modelB.clear()
        train_data = prs.parse('train2.wtag')
        modelB.train(train_data)
        modelB.comp('comp2.words')

