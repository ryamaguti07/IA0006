%%
clear
clc
M = csvread('dados_voz_genero.csv',1,1);
M = M(randperm(size(M, 1)), :);
percent=0.2;
ss=size(M(:,1));
t_ind = ceil(ss(1)*percent);
trainM=M(1:ss(1)-t_ind,:);
testM=M(ss(1)-t_ind:ss(1),:);

%%

%------------histogram-----------------------------
for i=1:20
    str='';
    switch i
        case 1
            str='sd';
        case 2
            str='median';
        case 3
            str='Q25';
        case 4
            str='Q75';
        case 5
            str='IQR';
        case 6
            str='skew';
        case 7
            str='kurt';
        case 8
           str='sp.ent';
        case 9
            str='sfm';
        case 10
            str='mode';
        case 11
            str='centroid';
        case 12
            str='meanfun';
        case 13
            str='minfun';
        case 14
            str='maxfun';
        case 15
            str='meandom';
        case 16
            str='mindom';           
        case 17
            str='maxdom';
        case 18
            str='dfrange';
        case 19
            str='modindx';   
    end
    figure(i)
    histogram(M(:,i))
    title(str)
end
%%
Mcor = M(:,1:19)
labels={'sd','median','Q25','Q75','IQR','skew','kurt','sp.ent','sfm','mode','centroid','meanfun','minfun','maxfun','meandom','mindom','maxdom','dfrange','modindx'}
for m1 = 1:size(Mcor,2); % Create correlations for each experimenter
 for m2 = 1:size(Mcor,2); % Correlate against each experimenter
  Cor(m2,m1) = corr(Mcor(:,m1),Mcor(:,m2));
 end
end
imagesc(Cor);
colormap(jet);
colorbar;


set(gca, 'XTick', 1:19); % center x-axis ticks on bins
set(gca, 'YTick', 1:19); % center y-axis ticks on bins
set(gca,'XTickLabel',labels);   % gca gets the current axis
set(gca,'YTickLabel',labels);   % gca gets the current axis


%%
%training
x_train = trainM(:,1:19);
x_label=trainM(:,20);

%compute the cost and gradient
[m n] = size(x_train);
theta0=zeros((n+1),1);
x_train = [ones(m, 1) x_train];
[J grad] = computeCost(theta0,x_train,x_label);
options = optimset('GradObj', 'on', 'MaxIter', 600); 
[theta, cost] =fminunc(@(t)(computeCost(t, x_train, x_label)), theta0, options);
%calculating accuracy on training
p = round(sigmoid(x_train * theta));


%%
%prediciting
x_test = testM(:,1:19);
test_label=testM(:,20);
[m n] = size(x_test);
x_test = [ones(m,1) x_test];
y = round(sigmoid(x_test * theta));
y1 = sigmoid(x_test * theta);
acc = mean(double(y==test_label));
%%
%metrics

confuM = confusionmat(y,test_label);
yLabel = categorical(y);
testLabel =categorical(test_label);
plotconfusion(yLabel,testLabel)
stats = confusionmatStats(test_label,y)
%%
[X,Y] = perfcurve(test_label,y,'1');
plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by Logistic Regression')

%%
tresh =0;
count=1;
while tresh<=1
    for i=1:635
        if(y1(i,1)>tresh)
            y2(i)=1;
        else
            y2(i)=0;
        end
    end
    stats1 = confusionmatStats(test_label,y2)
    treshScore(count,1)=tresh;
    treshScore(count,2)=stats1.Fscore(1);
    count = count+1;
    tresh=tresh+0.025;
end
%%
 plot(treshScore(:,1),treshScore(:,2))
 xlabel('Threshold') 
ylabel('F-1 Score')
title('Threshold x F-1 Score')