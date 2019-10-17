%%
clear all
clc
x_train=importdata('x_train.txt',' ');
train_labels=importdata('y_train.txt',' ');
x_train = normalize(x_train);
x_test=importdata('X_test.txt',' ');
test_labels=importdata('y_test.txt',' ');
x_test = normalize(x_test);


%%
for i=1:6
    x_train1 = x_train;
    x_test1 = x_test;
    train_label = train_labels;
    train_label(train_labels~=i)=0;
    train_label(train_labels==i)=1;
    [m n] = size(x_train);
    theta0=zeros((n+1),1);
    x_train1 = [ones(m, 1) x_train1];
    [J grad] = computeCost(theta0,x_train1,train_label,0.01);
  %  options = optimset('GradObj', 'on', 'MaxIter', 200); 
   % [theta, cost] =fminunc(@(t)(computeCost(t, x_train1, train_label)), theta0, options);
    for c=1:2000
         [theta grad] = computeCost(theta,x_train1,train_label,0.01);
    end
   
    [m n] = size(x_test);
    x_test1 = [ones(m,1) x_test1];
    y(:,i) = (sigmoid(x_test1 * theta));


end
%%
for i=1:size(y(:,1))
    [val,loc] = max(y(i,:));
    y_pred(i)=loc;
end
y_pred = (y_pred)';


%%

yLabel = categorical(test_labels);
testLabel =categorical(y_pred);
plotconfusion(yLabel,testLabel);
confM1 = confusionmat(y_pred,test_labels);

[c_matrix,Result,RefereceResult]= confusionA.getMatrix(test_labels,y_pred);

logisticF1 = Result.F1_score;
logisticAcc = Result.Accuracy;

%%
for k=1:75
    [predicted_labels] = KNN_(k,x_train,train_labels,x_test,test_labels);
    [c_matrix,Result,RefereceResult]= confusionA.getMatrix(test_labels,predicted_labels);
    ArrAcc(k,1)=k;
    ArrAcc(k,2)=Result.Accuracy;
    ArrF1(k,1)=k;
    ArrF1(k,2)=Result.F1_score;
end
%%
plot(ArrAcc(:,1),ArrAcc(:,2))
xlabel('K')
ylabel('Accuracy')
title('KNN Accuracy')

figure(2)

plot(ArrF1(:,1),ArrF1(:,2))
xlabel('K')
ylabel('F-1 Score')
title('KNN F-1 Score')

%%
[predicted_labels] = KNN_(8,x_train,train_labels,x_test,test_labels);
confM2 = confusionmat(predicted_labels,test_labels);
yLabel = categorical(test_labels);
testLabel =categorical(predicted_labels);
plotconfusion(yLabel,testLabel)
%%


plot(ArrF1(:,1),ArrF1(:,2))
hold on
plot(ArrF1,ones(size(ArrF1)) * logisticF1)
xlabel('K')
ylabel('F-1 Score')
title('KNN and Logistic F-1 Score per K')
legend('KNN','Logistic Regression')