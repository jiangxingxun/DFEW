import sklearn.metrics

def get_WAR(trues_te, pres_te):
    WAR  = sklearn.metrics.accuracy_score(trues_te, pres_te)
    return WAR

def get_UAR(trues_te, pres_te):
    cm = sklearn.metrics.confusion_matrix(trues_te, pres_te) 
    acc_per_cls = [ cm[i,i]/sum(cm[i]) for i in range(len(cm))]
    UAR = sum(acc_per_cls)/len(acc_per_cls)
    return UAR

def get_cm(trues_te, pres_te):
    cm = sklearn.metrics.confusion_matrix(trues_te, pres_te) 
    return cm
