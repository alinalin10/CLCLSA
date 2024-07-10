import argparse
import os

from networks.trainers.clcl_trainer import CLCLSA_Trainer, EV_Trainer


if __name__ == '__main__':
    # python main_clcl.py --data_folder=ROSMAP --hidden_dim=300 --num_epoch=2500
    # python main_clcl.py --data_folder=BRCA --hidden_dim=200 --num_epoch=2500
    parser = argparse.ArgumentParser()

    # dataset settings
    parser.add_argument('--data_folder', type=str, default="ROSMAP")
    parser.add_argument('--missing_rate', type=float, default=0.2)
    parser.add_argument('--exp', type=str, default="./exp")

    # model params
    parser.add_argument('--hidden_dim', type=str, default="300")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--step_size', type=int, default=500)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--prediction', type=str, default="64,32")
    parser.add_argument('--device', type=str, default="cpu")

    parser.add_argument('--lambda_cl', type=float, default=0.05)
    parser.add_argument('--lambda_co', type=float, default=0.02)

    parser.add_argument('--lambda_al', type=float, default=1.)

    # training params
    parser.add_argument('--num_epoch', type=int, default=2500)
    parser.add_argument('--test_inverval', type=int, default=50)

    args = parser.parse_args()
    params = vars(args)
    params['hidden_dim'] = [int(x) for x in params['hidden_dim'].split(",")]
    params['prediction'] = {i: [int(x) for x in params['prediction'].split(",")] for i in range(3)}
    cl_trainer = CLCLSA_Trainer(params)
    cl_trainer.train()
    cl_trainer.plot() 

 

    
    parser_sv = argparse.ArgumentParser()
    parser_sv.add_argument('--batch-size', type=int, default=-1, metavar='N', help='input batch size for training [default: 100]')
    parser_sv.add_argument('--epochs', type=int, default=3000, metavar='N', help='number of epochs to train [default: 500]')
    parser_sv.add_argument('--epochs_val', type=int, default=50)
    # parser.add_argument('--lambda-epochs', type=int, default=50, metavar='N', help='gradually increase the value of lambda from 0 to 1')
    parser_sv.add_argument('--lr', type=float, default=0.00001, metavar='LR', help='learning rate')
    parser_sv.add_argument('--cv', type=int, default=4)

    parser_sv.add_argument('--hidden_dim', type=str, default="2000")

    # change to ROSMAP
    parser_sv.add_argument('--data_path', type=str, default="ROSMAP")
    parser_sv.add_argument('--exp_save_path', type=str, default="exp")
    args_sv = parser_sv.parse_args()
    params_sv = vars(args_sv)
    # params['hidden_dim'] = [int(x) for x in params['hidden_dim'].split(",")]
    
    args_sv = parser_sv.parse_args()
    
    ev_trainer = EV_Trainer(args_sv)

    

    
    ev_trainer.train()
    ev_trainer.load_model()
    auc, sen, spec, f1, acc, preds, gts, us, u_a = ev_trainer.test(9999)

    print(auc)
    print(sen)
    print(spec)
    print(f1)
    print(acc)
    print(preds)
    print(gts)
    print(us) 
    print(max(us))
    print(min(us))
    print(u_a)
    

    # plot here
    
    


    