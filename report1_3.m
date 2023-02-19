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

%まず，読み込む画像リストを作成します．Positive, Negative別々に作成してから，結合してみましょう．
PosList=list(1:150);   
NegList=list(151:300); 
Training={PosList{:} NegList{:}};

% AlexNetを使います．net に学習済モデルがセットされます．
net = alexnet;

IM = [];
%画像をIMに格納
for i=1:150
    img = imread(PosList{i});
    reimg = imresize(img,net.Layers(1).InputSize(1:2)); 
    IM = cat(4,IM,reimg);
end
for i=1:150
    img = imread(NegList{i});
    reimg = imresize(img,net.Layers(1).InputSize(1:2)); 
    IM = cat(4,IM,reimg);
end

% activationsを利用して中間特徴量を取り出します．
dcnnf = activations(net,IM,'fc6');  
% squeeze関数で，ベクトル化します．
dcnnf = squeeze(dcnnf);
% L2ノルムで割って，L2正規化．
% 最終的な dcnnf を画像特徴量として利用します．
dcnnf = (dcnnf/norm(dcnnf))';
training_label = [ones(150,1); ones(150,1)*(-1)];

cv=5;
idx=[1:n];
accuracy=[];
data_pos = dcnnf(1:150,:);
data_neg = dcnnf(151:300,:);
% idx番目(idxはcvで割った時の余りがi-1)が評価データ
% それ以外は学習データ
for i=1:cv
  train_pos=data_pos(find(mod([1:150],5)~=(i-1)),:);
  eval_pos =data_pos(find(mod([1:150],5)==(i-1)),:);
  train_neg=data_neg(find(mod([1:150],5)~=(i-1)),:);
  eval_neg =data_neg(find(mod([1:150],5)==(i-1)),:);

  train_data = [train_pos;train_neg];
  eval=[eval_pos;eval_neg];

  train_label=[ones(120,1);ones(120,1)*(-1)];
  eval_label =[ones(30,1); ones(30,1)*(-1)];
  model = fitcsvm(train_data, train_label,'KernelFunction','linear');
  [predicted_label, scores] = predict(model, eval);
% 　ac = 評価(認識精度値を出力)
  ac = numel(find(eval_label==predicted_label))/numel(eval_label);
  accuracy=[accuracy ac];
end

disp(mean(accuracy));