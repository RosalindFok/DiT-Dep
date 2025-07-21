import gc
import re
import json
import torch
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from collections import defaultdict, OrderedDict

from metric import Metric
from dataset import Depression_Dataset
from model import device, DIT_DEP, Combined_Loss
from path import Brain_Atlas_Dir_Path, Running_File_Dir_Path
from config import seed, n_splits, Group, Experiment_Config, Mild_Config
from utils import fix_random, read_json, write_json, write_csv, plot_loss

def train(
    model,
    train_dataloader,
    modality_list,
    loss_fn,
    optimizer,
    fold : int,
    epoch : int,
    epochs : range,
    loss_dict : dict[str, list]
) -> None:
    model.train()
    train_loss_list = [] # train loss
    allocated_list, reserved_list, total_list = [], [], [] # GPU usage
    for batch in train_dataloader:
        # data
        input_dict = {
            k : getattr(batch, k).to(device) 
            for k in modality_list
        }
        target = getattr(batch, Experiment_Config.T).to(device).long().flatten()
        # forward
        output = model(input_dict=input_dict)
        # loss
        loss = loss_fn(input=output.logits, target=target, x_0=output.x_proj, x_p=output.x_pred)
        assert not torch.isnan(loss), f"Loss is NaN."
        train_loss_list.append(loss.item())
        # 3 steps of back propagation
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        # monitor GPU memory usage
        if torch.cuda.is_available() and epoch == epochs.start:
            allocated_list.append(torch.cuda.memory_allocated(device=device)/1024**3) # GB
            reserved_list.append(torch.cuda.memory_reserved(device=device)/1024**3) # GB
            total_list.append(torch.cuda.get_device_properties(device=device).total_memory/1024**3) # GB
    # GPU usage
    if torch.cuda.is_available() and epoch == epochs.start:
        logging.info(msg=f"GPU: Allocated {max(allocated_list):.2f} GB, Reserved {max(reserved_list):.2f} GB, Total {max(total_list):.2f} GB.")
    # logging loss
    train_loss = sum(train_loss_list)/len(train_loss_list)
    logging.info(msg=f"Fold {fold}, Epoch {epoch+1}/{epochs.stop}")
    logging.info(msg=f"Train Loss: {train_loss}")
    loss_dict[Experiment_Config.TRAIN].append(train_loss)

def test(
    model,
    test_dataloader,
    modality_list,
    loss_fn,
    loss_dict : dict[str, list],
    use_xai_metric : bool
) -> tuple[list[str], list[int], list[float], list[int], dict[str, float]]:
    model.eval()
    subj_list, prob_list, pred_list, true_list, valid_loss_list = [], [], [], [], []
    with torch.no_grad():
        for batch in test_dataloader:
            # data
            input_dict = {
                k : getattr(batch, k).to(device) 
                for k in modality_list
            }
            target = getattr(batch, Experiment_Config.T).to(device).long().flatten()
            # forward
            output = model(input_dict=input_dict)
            # loss
            loss = loss_fn(input=output.logits, target=target, x_0=output.x_proj, x_p=output.x_pred)
            assert not torch.isnan(loss), f"Loss is NaN."
            valid_loss_list.append(loss.item())
            # result
            subj_list.extend(getattr(batch, Experiment_Config.I))
            prob_list.extend(output.logits.softmax(dim=-1)[:, Group.DP].cpu().numpy())
            pred_list.extend(output.logits.argmax(dim=-1).cpu().numpy())
            true_list.extend(target.cpu().numpy())
    # logging loss
    valid_loss = sum(valid_loss_list) / len(valid_loss_list)
    logging.info(msg=f"Valid Loss: {valid_loss}")
    loss_dict[Experiment_Config.TEST].append(valid_loss)
    # logging metrics
    if not use_xai_metric: # binary classification
        metrics = {
            "AUC" : Metric.AUC(prob_list=prob_list, true_list=true_list),
            "ACC" : Metric.ACC(pred_list=pred_list, true_list=true_list),
            "PRE" : Metric.PRE(pred_list=pred_list, true_list=true_list),
            "SEN" : Metric.SEN(pred_list=pred_list, true_list=true_list),
            "F1S" : Metric.F1S(pred_list=pred_list, true_list=true_list),
        }  
    else:
        metrics = {
            "FPR" : Metric.FPR(pred_list=pred_list, true_list=true_list)
        }
    logging.info(msg=f"Test Metrics:\n{json.dumps(metrics, indent=4, ensure_ascii=False)}")

    return subj_list, true_list, prob_list, pred_list, metrics

def main() -> None:
    # fix the torch, numpy, random
    fix_random(seed=seed)

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--disorder", type=str, choices=[Experiment_Config.MILD])
    parser.add_argument("--method", type=str, choices=[
        # Optimal
        Experiment_Config.O, 
        # XAI
        Experiment_Config.PN, Experiment_Config.PE,
        # Ablation
        Experiment_Config.woDiT, Experiment_Config.woAtt, Experiment_Config.woTS, Experiment_Config.woFC, Experiment_Config.woMSE
    ])
    args = parser.parse_args()
    disorder_type = args.disorder
    method = args.method

    # logging 
    log_dir_root_path = Path("logs") / disorder_type
    log_dir_root_path.mkdir(parents=True, exist_ok=True)

    # config of the corresponding disorder
    if disorder_type == Experiment_Config.MILD:
        config = Mild_Config()
    else:
        raise NotImplementedError(f"Configuration for depression type '{disorder_type}' is not implemented.")

    # Node(time series), Edge(functional connectivity)
    modality_list = [Experiment_Config.TS, Experiment_Config.FC]
    if method in [Experiment_Config.woTS, Experiment_Config.PE]: # use edge, remove node
        modality_list.remove(Experiment_Config.TS)
        pth_suffix = "edge.pth"
    if method in [Experiment_Config.woFC, Experiment_Config.PN]: # use node, remove edge
        modality_list.remove(Experiment_Config.FC)
        pth_suffix = "node.pth"
    
    # Ablation on important modules
    use_dit = (method != Experiment_Config.woDiT)
    use_att = (method != Experiment_Config.woAtt)
    use_mse = (method != Experiment_Config.woMSE)

    # Identification of neuroimaging biomarkers
    xai_method_list = [Experiment_Config.PN, Experiment_Config.PE]
    if method in xai_method_list:
        # {Yeo_name : {net_name : net_labels}}
        experiment_dict = defaultdict(list)
        networks = read_json(json_path=Brain_Atlas_Dir_Path.network_json_path)["Yeo_17network"]
        for net_name, net_labels in networks.items():
            experiment_dict[re.split(pattern=r"[\s-]+", string=net_name)[0]].extend(net_labels)
        experiment_dict = {f"Yeo_Network_{method}" : experiment_dict}
        xai_dict = defaultdict(dict)
    else:
        # {"Whole_brain" : {variant_name : None}}
        experiment_dict = {"Whole_brain" : {method : None}}

    for exp_name, exp_dict in experiment_dict.items():
        for name, labels in exp_dict.items():
            log_dir_path = log_dir_root_path / exp_name / name.replace(" ", "_")
            log_dir_path.mkdir(parents=True, exist_ok=True)
            log_path = log_dir_path / "log.log" if len(exp_dict) == 1 else log_dir_root_path / exp_name / "log.log"
            logging.basicConfig(level=logging.INFO, format="%(message)s", filename=log_path, filemode="w")
            all_metrics = defaultdict(list)
            # Folds
            for fold in n_splits:
                # dataloader
                train_dataset = Depression_Dataset(depression_type=disorder_type, fold=fold, task=Experiment_Config.TRAIN, xai_method=method, net_labels=labels)
                test_dataset  = Depression_Dataset(depression_type=disorder_type, fold=fold, task=Experiment_Config.TEST,  xai_method=method, net_labels=labels)
                train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=config.num_workers)
                test_dataloader  = DataLoader(dataset=test_dataset,  batch_size=config.batch_size, shuffle=config.shuffle, num_workers=config.num_workers)

                # shape
                batch_data = next(iter(test_dataloader))
                fmri_shape = {
                    k : getattr(batch_data, k).shape[1:] 
                    for k in modality_list
                }
                logging.info(msg=f"Input shapes of {name}: {json.dumps(fmri_shape, indent=4, ensure_ascii=False)}")

                # model
                model = DIT_DEP(
                    shape_dict=fmri_shape, 
                    num_class=config.num_class, 
                    latent_dim=config.latent_dim,
                    use_dit=use_dit, use_att=use_att
                ).to(device=device)
                trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logging.info(msg=f"{model}")
                logging.info(msg=f"The number of trainable parameters is {trainable_parameters}.")
                
                # loss
                loss_fn = Combined_Loss(cet_weight=1.0, mse_weight=0.4, use_aux=(use_dit and use_mse))

                # optimizer
                optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.lr)

                # save model path
                pth_path = Running_File_Dir_Path.root_dir / "_".join([f"{fold}", pth_suffix]) if "pth_suffix" in locals() else None
                
                loss_dict = defaultdict(list)
                if method in xai_method_list: # no train
                    assert pth_path.exists(), f"please train the model with -woTS and -woFC"
                    model.load_state_dict(torch.load(pth_path, weights_only=True))
                for epoch in tqdm(config.epochs, desc=f"{name}. Fold {fold}", leave=False):
                    if method not in xai_method_list:
                        # learning rate decay
                        lr = config.lr*((1-epoch/config.epochs.stop)**0.8)
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr
                        # train
                        train(
                            model=model,
                            train_dataloader=train_dataloader,
                            modality_list=modality_list,
                            loss_fn=loss_fn,
                            optimizer=optimizer,
                            fold=fold,
                            epoch=epoch,
                            epochs=config.epochs,
                            loss_dict=loss_dict
                        )

                    # valid = test, to plot the loss
                    subj_list, true_list, prob_list, pred_list, metrics = test(
                        model=model,
                        test_dataloader=test_dataloader,
                        modality_list=modality_list,
                        loss_fn=loss_fn,
                        loss_dict=loss_dict,
                        use_xai_metric=(method in xai_method_list)
                    )
                
                # save the model for XAI
                if method in [Experiment_Config.woFC, Experiment_Config.woTS]:
                    torch.save(model.state_dict(), pth_path)

                # save the results of this fold 
                plot_loss(loss_dict=loss_dict, path=log_dir_path / f"fold_{fold}_loss.png")
                write_csv(
                    filename=log_dir_path / f"fold_{fold}_test_result.csv",
                    head=["subj",    "true",    "prob",    "pred"],
                    data=[ subj_list, true_list, prob_list, pred_list]
                )
                for key, value in metrics.items():
                    all_metrics[key].append(value)

                # clear
                del train_dataset, test_dataset, train_dataloader, test_dataloader, model, loss_fn, optimizer
                gc.collect()
                torch.cuda.empty_cache()
            
            # save the results of all folds
            averaged_metrics = {key : sum(value)/len(value) for key, value in all_metrics.items()}
            write_json(json_path=log_dir_path/"metric.json", dict_data={**{"mean":averaged_metrics}, **{"fold":all_metrics}})

            # XAI
            if method in xai_method_list:
                for key, value in averaged_metrics.items():
                    xai_dict[key][name] = value

        if method in xai_method_list:
            for key, value in xai_dict.items():
                value = OrderedDict(sorted(value.items(), key=lambda x : x[1]))
                xai_dict[key] = value
            write_json(json_path=log_dir_root_path / exp_name / "xai.json", dict_data=xai_dict)

if __name__ == "__main__":
    main()