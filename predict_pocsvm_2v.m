function [auc, auc1, auc2, auc2v, acc, acc1, acc2, acc2v] = predict_pocsvm_2v(model, x, x2, y)
[n,~] = size(x);

label1 = kernel(x, model.x, model.kerType, model.g4kerB) * model.Wa - model.rA; 
label2 = kernel(x2, model.x2, model.kerType, model.g4kerB) * model.Wb - model.rB;
label = (label1+label2)/2;

label1 = sign(label1);
label2 = sign(label2);
label = sign(label);

auc2v = roc_curve(y, label);
acc2v = length(find(y-label==0))/n; 
auc1 = roc_curve(y, label1);
acc1 =length(find(y-label1==0))/n;
auc2 = roc_curve(y, label2);
acc2=length(find(y-label2==0))/n;

auc=max([auc1,auc2,auc2v]);
acc=max([acc1,acc2,acc2v]);
end
