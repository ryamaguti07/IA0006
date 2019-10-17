function [prdistaictdista_labels] = KNN_(k,data,labels,t_data,t_labels)

prdistaictdista_labels=zeros(size(t_data,1),1);
dista=zeros(size(t_data,1),size(data,1));  
ind=zeros(size(t_data,1),size(data,1)); 
k_nn=zeros(size(t_data,1),k); 

for test_point=1:size(t_data,1)
    for train_point=1:size(data,1)
        dista(test_point,train_point)=sqrt(sum((t_data(test_point,:)-data(train_point,:)).^2));
    end
    [dista(test_point,:),ind(test_point,:)]=sort(dista(test_point,:));
end


k_nn=ind(:,1:k);
nn_index=k_nn(:,1);

%majority vote
for i=1:size(k_nn,1)
    options=unique(labels(k_nn(i,:)'));
    max_count=0;
    max_label=0;
    for j=1:length(options)
        L=length(find(labels(k_nn(i,:)')==options(j)));
        if L>max_count
            max_label=options(j);
            max_count=L;
        end
    end
    prdistaictdista_labels(i)=max_label;
end
