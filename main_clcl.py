import argparse
import os

from networks.trainers.clcl_trainer import CLCLSA_Trainer_3, CLCLSA_Trainer_2, EV_Trainer


if __name__ == '__main__':

    



    # SINGLE VIEW
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
    
    print("1 VIEW")
    ev_trainer.train()
    ev_trainer.load_model()
    acc_sv, us_sv = ev_trainer.get_lists()
    #print(acc_sv)
    print(f"length of acc_sv: {len(acc_sv)}")
    #u_a_sv = [tensor.item() for tensor in u_a_sv]
    #u_a_sv = u_a_sv.detach().numpy().flatten()
    
    #print(u_a_sv)
    print(f"length of u_a_sv: {len(us_sv)}")
    # u_a_sv length: 6360
    auc, sen, spec, f1, acc, preds, gts, us, u_a = ev_trainer.test(9999)

    """
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
    #u_a = [tensor.item() for tensor in u_a]
    #print(u_a)
    """
    # 106 length of u_a
    


    
    # 2 VIEW

    # python main_clcl.py --data_folder=ROSMAP --hidden_dim=300 --num_epoch=2500
    # python main_clcl.py --data_folder=BRCA --hidden_dim=200 --num_epoch=2500
    parser_2v = argparse.ArgumentParser()

    # dataset settings
    parser_2v.add_argument('--data_folder', type=str, default="ROSMAP")
    parser_2v.add_argument('--missing_rate', type=float, default=0.2)
    parser_2v.add_argument('--exp', type=str, default="./exp")

    # model params
    # originally 300
    parser_2v.add_argument('--hidden_dim', type=str, default="300")
    parser_2v.add_argument('--lr', type=float, default=1e-4)
    parser_2v.add_argument('--step_size', type=int, default=500)
    parser_2v.add_argument('--dropout', type=float, default=0.5)
    parser_2v.add_argument('--prediction', type=str, default="64,32")
    parser_2v.add_argument('--device', type=str, default="cpu")

    parser_2v.add_argument('--lambda_cl', type=float, default=0.05)
    parser_2v.add_argument('--lambda_co', type=float, default=0.02)

    parser_2v.add_argument('--lambda_al', type=float, default=1.)

    # training params
    parser_2v.add_argument('--num_epoch', type=int, default=2500)
    parser_2v.add_argument('--test_inverval', type=int, default=50)

    args_2v = parser_2v.parse_args()
    params_2v = vars(args_2v)
    params_2v['hidden_dim'] = [int(x) for x in params_2v['hidden_dim'].split(",")]
    params_2v['prediction'] = {i: [int(x) for x in params_2v['prediction'].split(",")] for i in range(3)}
    cl_trainer_2v = CLCLSA_Trainer_2(params_2v)
    print("2 VIEW")
    cl_trainer_2v.train()
    cl_trainer_2v.plot()

    # store important values 

    preds_2v, gts_2v, us_2v, acc_2v = cl_trainer_2v.testing()



    # 3 VIEW

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
    parser.add_argument('--device', type=str, default="cpu") # changed to cpu

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
    cl_trainer = CLCLSA_Trainer_3(params)
    print("3 VIEW")
    cl_trainer.train() 
    cl_trainer.plot()

    preds_3v, gts_3v, us_3v, acc_3v = cl_trainer.testing()
    
    print(f"length of acc_2v: {len(acc_2v)}")
    print(f"length of us_2v: {len(us_2v)}")
    print(f"length of acc_3v: {len(acc_3v)}")
    print(f"length of us_3v: {len(us_3v)}")

    # finding optimal threshold
    # TODO find T2 first (single view is more difficult to extract data do that after this function works with T2)
    # got rid of single view results
    """
    def find_optimal_threshold(results_sv, results_2v, results_3v):
        def extract_accuracy_and_uncertainty(results):
            accuracy = results['accuracy']
            uncertainty = results['uncertainty']
            return accuracy, uncertainty

        acc_sv, unc_sv = extract_accuracy_and_uncertainty(results_sv)
        acc_2v, unc_2v = extract_accuracy_and_uncertainty(results_2v)
        acc_3v, unc_3v = extract_accuracy_and_uncertainty(results_3v)

        def find_threshold(acc_1, unc_1, acc_2, unc_2):
            best_threshold = None
            min_difference = float('inf')
            # map uncertainty values to accuracy values
            # each accuracy corresponds to step uncertainty values
            step = len(unc_1) // len(acc_1) 
            for i in range(len(acc_1)):
                unc_segment_1 = unc_1[i * step: (i + 1) * step]
                unc_segment_2 = unc_2[i * step: (i + 1) * step]
                
                avg_unc_1 = sum(unc_segment_1) / len(unc_segment_1)

                # i dont think this is neccessary
                avg_unc_2 = sum(unc_segment_2) / len(unc_segment_2)
                
                acc_diff = abs(acc_2[i] - acc_1[i])
                
                if acc_diff < min_difference:
                    min_difference = acc_diff
                    best_threshold = avg_unc_1
            return best_threshold

        # find optimal thresholds where accuracy improvement is minimal
        optimal_T1 = find_threshold(acc_sv, unc_sv, acc_2v, unc_2v)
        optimal_T2 = find_threshold(acc_2v, unc_2v, acc_3v, unc_3v)

        # took out optimal_T1
        return optimal_T1, optimal_T2
    
    us_2v = [tensor.item() for tensor in us_2v]
    us_3v = [tensor.item() for tensor in us_3v]

    # dictionaries w acc and unc
    results_sv = {
        'accuracy': acc_sv,
        'uncertainty': u_a
    }
    results_2v = {
        'accuracy': acc_2v,
        'uncertainty': us_2v
    }
    results_3v = {
        'accuracy': acc_3v,
        'uncertainty': us_3v
    }

    optimal_T1, optimal_T2 = find_optimal_threshold(results_sv, results_2v, results_3v)
    print(f"Optimal T1: {optimal_T1}, Optimal T2: {optimal_T2}")
    print(f"acc2max: {max(acc_2v)}")
    print(f"us2max: {max(us_2v)}")
    print(f"acc3max: {max(acc_3v)}")
    print(f"us3max: {max(us_3v)}")
    """

    def find_optimal_threshold(results_sv, results_2v, results_3v):
        def extract_accuracy_and_uncertainty(results):
            accuracy = results['accuracy']
            uncertainty = results['uncertainty']
            return accuracy, uncertainty

        acc_sv, unc_sv = extract_accuracy_and_uncertainty(results_sv)
        acc_2v, unc_2v = extract_accuracy_and_uncertainty(results_2v)
        acc_3v, unc_3v = extract_accuracy_and_uncertainty(results_3v)

        def find_threshold(acc_1, unc_1, acc_2, unc_2):
            best_threshold = None
            min_difference = float('inf')
            # Calculate the step size
            step_1 = len(unc_1) // len(acc_1) 
            step_2 = len(unc_2) // len(acc_2)

            for i in range(len(acc_1)):
                start_idx_1 = i * step_1
                end_idx_1 = start_idx_1 + step_1
                start_idx_2 = i * step_2
                end_idx_2 = start_idx_2 + step_2

                unc_segment_1 = unc_1[start_idx_1:end_idx_1]
                unc_segment_2 = unc_2[start_idx_2:end_idx_2]

                # Ensure segments are not empty to avoid division by zero
                if len(unc_segment_1) == 0 or len(unc_segment_2) == 0:
                    continue

                avg_unc_1 = sum(unc_segment_1) / len(unc_segment_1)

                # I don't think this is necessary
                avg_unc_2 = sum(unc_segment_2) / len(unc_segment_2)
                
                acc_diff = abs(acc_2[i] - acc_1[i])
                
                if acc_diff < min_difference:
                    min_difference = acc_diff
                    best_threshold = avg_unc_1
            return best_threshold

        # Find optimal thresholds where accuracy improvement is minimal
        optimal_T1 = find_threshold(acc_sv, unc_sv, acc_2v, unc_2v)
        optimal_T2 = find_threshold(acc_2v, unc_2v, acc_3v, unc_3v)

        # Return optimal thresholds
        return optimal_T1, optimal_T2

    # Assuming us_2v and us_3v are tensors, convert them to lists
    us_2v = [tensor.item() for tensor in us_2v]
    us_3v = [tensor.item() for tensor in us_3v]

    # Dictionaries with accuracy and uncertainty
    results_sv = {
        'accuracy': acc_sv,
        'uncertainty': us_sv
    }
    results_2v = {
        'accuracy': acc_2v,
        'uncertainty': us_2v
    }
    results_3v = {
        'accuracy': acc_3v,
        'uncertainty': us_3v
    }

    optimal_T1, optimal_T2 = find_optimal_threshold(results_sv, results_2v, results_3v)
    print(f"Optimal T1: {optimal_T1}, Optimal T2: {optimal_T2}")
    print(f"acc2max: {max(acc_2v)}")
    print(f"us2max: {max(us_2v)}")
    print(f"acc3max: {max(acc_3v)}")
    print(f"us3max: {max(us_3v)}")


    

    
    


    


