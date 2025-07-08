import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

from config import Group

class Metric:
    @staticmethod
    def AUC(prob_list : list[float], true_list : list[int]) -> float:
        prob = np.array(object=prob_list).astype(dtype=float)
        true = np.array(object=true_list).astype(dtype=int)
        assert prob.shape == true.shape, f"Invalid shape: prob={prob.shape}, true={true.shape}"
        return float(roc_auc_score(y_true=true, y_score=prob))
    
    @staticmethod
    def ACC(pred_list : list[int], true_list : list[int]) -> float:
        pred = np.array(object=pred_list).astype(dtype=int)
        true = np.array(object=true_list).astype(dtype=int)
        assert pred.shape == true.shape, f"Invalid shape: pred={pred.shape}, true={true.shape}"
        return float(accuracy_score(y_true=true, y_pred=pred))
    
    @staticmethod
    def PRE(pred_list : list[int], true_list : list[int]) -> float:
        pred = np.array(object=pred_list).astype(dtype=int)
        true = np.array(object=true_list).astype(dtype=int)
        assert pred.shape == true.shape, f"Invalid shape: pred={pred.shape}, true={true.shape}"
        return float(precision_score(y_true=true, y_pred=pred, zero_division=0))

    @staticmethod
    def SEN(pred_list : list[int], true_list : list[int]) -> float:
        pred = np.array(object=pred_list).astype(dtype=int)
        true = np.array(object=true_list).astype(dtype=int)
        assert pred.shape == true.shape, f"Invalid shape: pred={pred.shape}, true={true.shape}"
        return float(recall_score(y_true=true, y_pred=pred, zero_division=0))

    @staticmethod
    def F1S(pred_list : list[int], true_list : list[int]) -> float:
        pred = np.array(object=pred_list).astype(dtype=int)
        true = np.array(object=true_list).astype(dtype=int)
        assert pred.shape == true.shape, f"Invalid shape: pred={pred.shape}, true={true.shape}"
        return float(f1_score(y_true=true, y_pred=pred))
    
    @staticmethod
    def POS_ACC(pred_list : list[int], true_list : list[int]) -> float:
        pred = np.array(object=pred_list).astype(dtype=int)
        true = np.array(object=true_list).astype(dtype=int)
        assert pred.shape == true.shape, f"Invalid shape: pred={pred.shape}, true={true.shape}"
        mask = (true == Group.DP)
        return float(accuracy_score(y_true=true[mask], y_pred=pred[mask]))

    @staticmethod
    def NEG_ACC(pred_list : list[int], true_list : list[int]) -> float:
        pred = np.array(object=pred_list).astype(dtype=int)
        true = np.array(object=true_list).astype(dtype=int)
        assert pred.shape == true.shape, f"Invalid shape: pred={pred.shape}, true={true.shape}"
        mask = (true == Group.HC)
        return float(accuracy_score(y_true=true[mask], y_pred=pred[mask]))
