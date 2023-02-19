n=0; list={};
LIST={'gyoza','karaage'};
DIR0="./";
for i=1:length(LIST)
    DIR=strcat(DIR0,LIST(i),'/')
    W=dir(DIR{:});
    for j=1:size(W)
      if (strfind(W(j).name,'.jpg'))
        fn=strcat(DIR{:},W(j).name);
        n=n+1;
        fprintf('[%d] %s\n',n,fn);
        list={list{:} fn};
      end
    end
end
    
n=300;
n_bof = 1000;
bof = zeros(n,n_bof);

%まず，読み込む画像リストを作成します
PosList=list(1:150);   
NegList=list(151:300); 
Training={PosList{:} NegList{:}};

% 次に forループで，全画像についてSURF特徴を抽出します．
Features=[];
for i=1:300
  I=rgb2gray(imread(Training{i}));
  p=detectSURFFeatures(I);
  [f,p2]=extractFeatures(I,p);
  Features=[Features; f];
end
[idx,codebook]=kmeans(Features,n_bof);

for j=1:n
    I=rgb2gray(imread(Training{j}));
    p=detectSURFFeatures(I);
    [f,p2]=extractFeatures(I,p);  
    for i=1:size(p2,1)
        now_feature = p2(i).Metric;
        index = -1;
        min_value = power(10000,10);        
        for ci=1:n_bof
            temp = 0;
            for cj=1:64
                temp = temp + power(codebook(ci,cj)-now_feature,2);
            end
            if temp < min_value
                min_value = temp;
                index = ci;
            end
        end
        bof(j,index)=bof(j,index)+1;
    end
end
disp("評価");
data_pos = bof(1:150,:);
data_neg = bof(151:300,:);
accuracy = [];
for i=1:5
    train_pos=data_pos(find(mod([1:150],5)~=(i-1)),:);
    eval_pos =data_pos(find(mod([1:150],5)==(i-1)),:);
    train_neg=data_neg(find(mod([1:150],5)~=(i-1)),:);
    eval_neg =data_neg(find(mod([1:150],5)==(i-1)),:);

    train_data = [train_pos;train_neg];
    eval=[eval_pos;eval_neg];
    train_label=[ones(120,1);ones(120,1)*(-1)];
    eval_label =[ones(30,1); ones(30,1)*(-1)];

    model_rbf = fitcsvm(train_data,train_label,'KernelFunction','rbf','KernelScale','auto');
    [predicted_label, scores] = predict(model_rbf, eval);
    ac = numel(find(eval_label==predicted_label))/numel(predicted_label);
    accuracy = [accuracy, ac];
end

disp(mean(accuracy));