import sys, os
from datetime import datetime
import time
import engine, utility  

if __name__ == '__main__' :
    if len(sys.argv) <=1:
        raise Exception('configure file must be specified!')

    args = utility.load_params(json_file = sys.argv[1])
    checkpoint_dir = args['checkpoint_path'] + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    utility.save_args(checkpoint_dir, args)
    logger = utility.Tee(checkpoint_dir + '/log.txt', 'a')

    description = args['description']
    print('Experiment is running!')
    print('-----------------------------------------------------------------------------')
    print('Model: ' + args['model']['network'])
    print('Optimizer: ' + args['model']['optimizer'])
    print('Batch Size: ' + str(args['data']['train']['batch_size']))
    print('Description: ' + description)
    print('-----------------------------------------------------------------------------')
    

    os.environ['CUDA_VISIBLE_DEVICES'] = args['model']['device_ids']

    runner = engine.Engine(args, checkpoint_dir)
    test_freq = args['model']['test_freq']
    num_epoches = args['model']['sub_epoches']

    for e in range(runner.epoch, num_epoches):
        for s in args['model']['levels'] :
            runner.train_PaDNet(mode = s)
            if e % test_freq == 0:
                runner.validate_PaDNet(mode = s)

    num_epoches = args['model']['epoches']


    for s in args['model']['levels']:
        runner.load_parameters(s)


    for e in range(runner.epoch, num_epoches):
        runner.train_PaDNet(mode = 'total')
        if e % test_freq == 0:
            runner.validate_PaDNet(mode = 'total')
