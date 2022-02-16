clear, close all

X = xlsread('heart2.xlsx');
%o2 = xlsread('o2Saturation.xlsx');

%size(X)
%size(o2)

X((find(X(:,12)==4)),:) = []; %Remove null values from caa
X((find(X(:,13)==0)),:) = []; %Remove null values from thall
%X((find(X(:,10)==0)),:) = []; %Remove null values from oldpeak - not good
maxChol = max(X(:,5));
%figure, histogram(X(:,5)), title('chol')
%X(X(:,5)>=maxChol, :) = []; %Remove outlier from chol -UNdone
[N,n] = size(X);
%Remove unknown variable -UNdone
%X(:,10) = [];
%min(X(:,10))
%X(:,10)

T = array2table(X,...
    'VariableNames',{'age','sex','cp','trtbps','chol','fbs','restecg',...
    'thalachh','exng','oldpeak','slp','caa','thall','output'});
head(T)
oldpeak = X(:,10);
oldpeak((oldpeak==0)) = [];
%mean(oldpeak)
%min(oldpeak)
%max(oldpeak)
%std(oldpeak)
%median(oldpeak)

figure, histogram(X(:,10)), title('Old peaks')
figure, histogram(oldpeak), title('Removed null values')
%
% Mean imputation of missing values in oldpeak
X(X(:,10)==0 ,10) = mean(oldpeak);
figure, histogram(X(:,10)), title('Imputated means')
%
Classes = X(:,[2:3 6:7 9 11:14]);   
Floats = X(:, [1 4:5 8 10]);

Sex = Classes(:,1); %(1 = male; 0 = female)
cp = Classes(:,2); %Chest pain type (0 = typical angina; 1 = atypical angina; 2 = non-anginal pain; 3 = asymptomatic)
FBS = Classes(:,3); %(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
restecg = Classes(:,4); %resting electrocardiographic results (0 = normal; 1 = having ST-T; 2 = hypertrophy)
exang = Classes(:,5); %exercise induced angina (1 = yes; 0 = no)
Slope = Classes(:,6); %the slope of the peak exercise ST segment (0 = upsloping; 1 = flat; 2 = downsloping)
caa = Classes(:,7); %number of major vessels (0-3) colored by flourosopy
thal = Classes(:,8); % 2 = normal; 1 = fixed defect; 3 = reversable defect
num = Classes(:,9); %diagnosis of heart disease (angiographic disease status)
                    %0= less chance of heart attack 1= more chance of heart attack

age = Floats(:,1);
trtbps = Floats(:,2); %resting blood pressure (in mm Hg on admission to the hospital)
chol = Floats(:,3); %serum cholestoral in mg/dl
thalach = Floats(:,4); %maximum heart rate achieved
oldpeak = Floats(:,5); %ST depression induced by exercise relative to rest

%min(X(:,13))
%max(X(:,13))
%X(:,11)
%size(find(X(:,12)==0))
%I = find(X(:,11)==4)

%size(X)
%X(252,11)
% columns:
% |age|sex|cp|trtbps|chol|fbs|restecg|thalachh|exng|oldpeak|slp|caa|thall|

%Remove predictor class (target variable)
Data = X(:, 1:13);
% Keep in original
%X = X(:, 1:13);

%Z = zscore(X);
% Scaling Whole data to [-1,1]
minX = min(Data); maxX = max(Data); cofs = 2./(maxX-minX);
Data = bsxfun(@times,Data,cofs); Data = bsxfun(@minus,Data,(maxX+minX)./(maxX-minX));

% Scaling continuous values to [-1, 1]
minX = min(Floats); maxX = max(Floats); cofs = 2./(maxX-minX);
Floats = bsxfun(@times,Floats,cofs); Floats = bsxfun(@minus,Floats,(maxX+minX)./(maxX-minX));

%fix great outliers? -UNdone
figure, plotmatrix(Floats), title('Plotmatrix of continuous data')
figure, corrplot(Floats, 'varNames', {'age','trtbps','chol','thalachh','oldpeak'}), title('Correlation matrix of continuous data')

%min(Floats)
%max(Floats)
% Remove distorting variable
Floats(:,5) = [];
Data(:,10) = [];
%X(:,10) = []; - keep in original data

[mappedX, mapping] = compute_mapping(Floats, 'PCA', 2);

markerstr = 'x+*o';

Colors = lines(size(unique(num)));
figure, gscatter(mappedX(:,1), mappedX(:,2),num,Colors,markerstr,5.8),...
    legend('<50% chance of heart attack', '>50% chance of heart attack');
title('Diagnosed heart disease')
xlabel('PC1')
ylabel('PC2')

%size(unique(num))
%maxClasses = size(unique(Classes))

% figure
% Colors = lines(size(unique(num)));
% subplot(1,3,1), gscatter(mappedX(:,1), mappedX(:,2),num,Colors,markerstr,5.8),...
%     legend('<50% chance of heart attack', '>50% chance of heart attack');
% title('Diagnosed heart disease')

figure
Colors = lines(size(unique(Sex)));
%markerstr = 'x+*o';
subplot(2,2,1), gscatter(mappedX(:,1), mappedX(:,2),Sex,Colors,markerstr,5.8),...
    legend('Female', 'Male');
title('Sex')
xlabel('PC1')
ylabel('PC2')

Colors = lines(size(unique(cp)));
%markerstr = 'x+*o';
subplot(2,2,2), gscatter(mappedX(:,1), mappedX(:,2),cp,Colors,markerstr,5.8),...
    legend('Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic');
title('Chest pain type')
xlabel('PC1')
ylabel('PC2')
%
x = categorical({'Female','Male'});
fem = X((X(:,2)==0),:);
[fem0,~] = size(fem(fem(:,14)==0,:));
[fem1,~] = size(fem(fem(:,14)==1,:));
male = X((X(:,2)==1),:);
[male0,~] = size(male(male(:,14)==0,:));
[male1,~] = size(male(male(:,14)==1,:));
subplot(2,2,3), bar(x, [fem0 fem1; male0 male1]), legend('No disease', 'Disease'),...
    title('Heart disease frequency for sex')

x = categorical({'Typical angina','Atypical angina', 'Non-anginal pain', 'Asymptomatic'});
x = reordercats(x,{'Typical angina','Atypical angina', 'Non-anginal pain', 'Asymptomatic'});
typ = X((X(:,3)==0),:);
[typ0,~] = size(typ(typ(:,14)==0,:));
[typ1,~] = size(typ(typ(:,14)==1,:));
atyp = X((X(:,3)==1),:);
[atyp0,~] = size(atyp(atyp(:,14)==0,:));
[atyp1,~] = size(atyp(atyp(:,14)==1,:));
nap = X((X(:,3)==2),:);
[nap0,~] = size(nap(nap(:,14)==0,:));
[nap1,~] = size(nap(nap(:,14)==1,:));
asym = X((X(:,3)==3),:);
[asym0,~] = size(asym(asym(:,14)==0,:));
[asym1,~] = size(asym(asym(:,14)==1,:));
subplot(2,2,4), bar(x, [typ0 typ1; atyp0 atyp1; nap0 nap1; asym0 asym1]), legend('No disease', 'Disease'),...
    title('Heart disease frequency for chest pain')
%
%-----------------------------------------------------------------------------------------
figure
% Colors = lines(size(unique(num)));
% subplot(1,3,1), gscatter(mappedX(:,1), mappedX(:,2),num,Colors,markerstr,5.8),...
%     legend('<50% chance of heart attack', '>50% chance of heart attack');
% title('Diagnosed heart disease')

Colors = lines(size(unique(FBS)));
subplot(2,2,1), gscatter(mappedX(:,1), mappedX(:,2),FBS,Colors,markerstr,5.8),...
    legend('False', 'True');
title('Fasting blood sugar > 120 mg/dl')
xlabel('PC1')
ylabel('PC2')

Colors = lines(size(unique(restecg)));
subplot(2,2,2), gscatter(mappedX(:,1), mappedX(:,2),restecg,Colors,markerstr,5.8),...
    legend('Normal', 'Having ST-T', 'Hypertrophy');
title('Resting electrocardiographic results')
xlabel('PC1')
ylabel('PC2')
%
%
x = categorical({'False','True'});
f = X((X(:,6)==0),:);
[f0,~] = size(f(f(:,14)==0,:));
[f1,~] = size(f(f(:,14)==1,:));
t = X((X(:,6)==1),:);
[t0,~] = size(t(t(:,14)==0,:));
[t1,~] = size(t(t(:,14)==1,:));
subplot(2,2,3), bar(x, [f0 f1; t0 t1]), legend('No disease', 'Disease'),...
    title('Heart disease frequency for fbs')
%

x = categorical({'Normal','Having ST-T','Hypertrophy'});
x = reordercats(x,{'Normal','Having ST-T','Hypertrophy'});
nor = X((X(:,7)==0),:);
[nor0,~] = size(nor(nor(:,14)==0,:));
[nor1,~] = size(nor(nor(:,14)==1,:));
stt = X((X(:,7)==1),:);
[stt0,~] = size(stt(stt(:,14)==0,:));
[stt1,~] = size(stt(stt(:,14)==1,:));
hyp = X((X(:,7)==2),:);
[hyp0,~] = size(hyp(hyp(:,14)==0,:));
[hyp1,~] = size(hyp(hyp(:,14)==1,:));
subplot(2,2,4), bar(x, [nor0 nor1; stt0 stt1; hyp0 hyp1]), legend('No disease', 'Disease'),...
    title('Heart disease frequency for restecg')
%
%-----------------------------------------------------------------------------------------
figure
% Colors = lines(size(unique(num)));
% subplot(1,3,1), gscatter(mappedX(:,1), mappedX(:,2),num,Colors,markerstr,5.8),...
%     legend('<50% chance of heart attack', '>50% chance of heart attack');
% title('Diagnosed heart disease')

Colors = lines(size(unique(exang)));
subplot(2,2,1), gscatter(mappedX(:,1), mappedX(:,2),exang,Colors,markerstr,5.8),...
    legend('No', 'Yes');
title('Exercise induced angina')
xlabel('PC1')
ylabel('PC2')

Colors = lines(size(unique(Slope)));
subplot(2,2,2), gscatter(mappedX(:,1), mappedX(:,2),Slope,Colors,markerstr,5.8),...
    legend('Upsloping', 'Flat', 'Downsloping');
title('The slope of the peak exercise ST segment')
xlabel('PC1')
ylabel('PC2')
%
%
x = categorical({'No','Yes'});
f = X((X(:,9)==0),:);
[f0,~] = size(f(f(:,14)==0,:));
[f1,~] = size(f(f(:,14)==1,:));
t = X((X(:,9)==1),:);
[t0,~] = size(t(t(:,14)==0,:));
[t1,~] = size(t(t(:,14)==1,:));
subplot(2,2,3), bar(x, [f0 f1; t0 t1]), legend('No disease', 'Disease'),...
    title('Heart disease frequency for exng')
%

x = categorical({'Upsloping','Flat','Downsloping'});
x = reordercats(x,{'Upsloping','Flat','Downsloping'});
nor = X((X(:,11)==0),:);
[nor0,~] = size(nor(nor(:,14)==0,:));
[nor1,~] = size(nor(nor(:,14)==1,:));
stt = X((X(:,11)==1),:);
[stt0,~] = size(stt(stt(:,14)==0,:));
[stt1,~] = size(stt(stt(:,14)==1,:));
hyp = X((X(:,11)==2),:);
[hyp0,~] = size(hyp(hyp(:,14)==0,:));
[hyp1,~] = size(hyp(hyp(:,14)==1,:));
subplot(2,2,4), bar(x, [nor0 nor1; stt0 stt1; hyp0 hyp1]), legend('No disease', 'Disease'),...
    title('Heart disease frequency for slope')
%
%-----------------------------------------------------------------------------------------
figure
% Colors = lines(size(unique(num)));
% subplot(1,3,1), gscatter(mappedX(:,1), mappedX(:,2),num,Colors,markerstr,5.8),...
%     legend('<50% chance of heart attack', '>50% chance of heart attack');
% title('Diagnosed heart disease')

Colors = lines(size(unique(caa)));
subplot(2,2,1), gscatter(mappedX(:,1), mappedX(:,2),caa,Colors,markerstr,5.8),...
    legend('0', '1','2','3');
title('Number of major vessels colored by fluoroscopy')
xlabel('PC1')
ylabel('PC2')

Colors = lines(size(unique(thal)));
subplot(2,2,2), gscatter(mappedX(:,1), mappedX(:,2),thal,Colors,markerstr,5.8),...
    legend('Normal', 'Fixed defect', 'Reversable defect');
title('Thallium stress result')
xlabel('PC1')
ylabel('PC2')

%
x = categorical({'0','1', '2', '3'});
x = reordercats(x,{'0','1', '2', '3'});
typ = X((X(:,12)==0),:);
[typ0,~] = size(typ(typ(:,14)==0,:));
[typ1,~] = size(typ(typ(:,14)==1,:));
atyp = X((X(:,12)==1),:);
[atyp0,~] = size(atyp(atyp(:,14)==0,:));
[atyp1,~] = size(atyp(atyp(:,14)==1,:));
nap = X((X(:,12)==2),:);
[nap0,~] = size(nap(nap(:,14)==0,:));
[nap1,~] = size(nap(nap(:,14)==1,:));
asym = X((X(:,12)==3),:);
[asym0,~] = size(asym(asym(:,14)==0,:));
[asym1,~] = size(asym(asym(:,14)==1,:));
subplot(2,2,3), bar(x,[typ0 typ1; atyp0 atyp1; nap0 nap1; asym0 asym1]), legend('No disease', 'Disease'),...
    title('Heart disease frequency for colored vessels')
%

%
x = categorical({'Normal','Fixed defect', 'Reversable defect'});
x = reordercats(x,{'Normal','Fixed defect', 'Reversable defect'});
typ = X((X(:,13)==1),:);
[typ0,~] = size(typ(typ(:,14)==0,:));
[typ1,~] = size(typ(typ(:,14)==1,:));
atyp = X((X(:,13)==2),:);
[atyp0,~] = size(atyp(atyp(:,14)==0,:));
[atyp1,~] = size(atyp(atyp(:,14)==1,:));
nap = X((X(:,13)==3),:);
[nap0,~] = size(nap(nap(:,14)==0,:));
[nap1,~] = size(nap(nap(:,14)==1,:));
subplot(2,2,4), bar(x,[typ0 typ1; atyp0 atyp1; nap0 nap1]), legend('No disease', 'Disease'),...
    title('Heart disease frequency for thall')
%
%figure, scatter(age, trtbps)
%size(Data)
%size(num)
Xinput = X;
Xinput(:,14) = [];

DT = fitctree(Xinput, num, 'optimizehyperparameters', 'auto');
%DT = fitctree(X, num);

view(DT, 'mode', 'graph');
view(DT)

% How well DT predicts?
predictionDT = DT.predict(Xinput);
misclassifiedDT = ~(predictionDT == num);
disp([char('Misclassified observations with decision tree: '),num2str((sum(misclassifiedDT) / N)*100),char('%.')])
[confusionMatrixDT,orderDT] = confusionmat(num,predictionDT)
figure, plotconfusion(num', predictionDT'), title('Decision tree confusion matrix')
disp('------------------------------------------------------------')


% 10-fold cross validation for kNN

kfolds = 10;
c = cvpartition(num,'KFold',kfolds)
% - For each k in [5,100], sum the number of missclassified observations over 10 test sets
kvals = 1:100;
SumMissClassified = [];
for k = kvals
    s = 0;
    for foldi = 1:kfolds
        PredClasses = [];
        testFoldSize = sum(c.test(foldi)); 
        Xtest = Data(c.test(foldi),:); Xtrain = Data(c.training(foldi),:);
        TrueClassLabelsTest = num(c.test(foldi));
        TrueClassLabelsTrain = num(c.training(foldi));
        % kNN classification for the test fold
        for i=1:testFoldSize
            % Distances with respect to trainning data
            Dists = pdist2(Xtest(i,:),Xtrain); 
            [~,I] = sort(Dists);
            kLabels = TrueClassLabelsTrain(I(1:k));
            PredClass = mode(kLabels);
            PredClasses = [PredClasses; PredClass];
        end
        s = s + sum(PredClasses ~= TrueClassLabelsTest);
    end
    SumMissClassified = [SumMissClassified; s];
end
% - What are good k values that generalizes the kNN classifier well?
%
figure, plot(kvals,SumMissClassified), title('10-fold cross validation result')
%
%mean(SumMissClassified)
%std(SumMissClassified)
xlabel('k'); ylabel('#missclassifications');
%
% kNN classification

k = kvals(SumMissClassified == min(SumMissClassified));
if (~isscalar(k)) k = k(1,1); end
disp([char('K: '),num2str(k)])
[I, d] = knnsearch(Data,Data,'K',k);
kLabels = num(I);
predsKNN = mode(kLabels');
%preds'
misclassifiedKNN = ~(predsKNN' == num);
disp([char('Misclassified observations with kNN: '),num2str((sum(misclassifiedKNN) / N)*100),char('%.')])
[confusionMatrixKNN,orderKNN] = confusionmat(num,predsKNN')
figure, plotconfusion(num', predsKNN), title('kNN confusion matrix')
disp('------------------------------------------------------------')


% NB classifier

NB = fitcnb(Data,num);
predictionNB = NB.predict(Data); 
%misclassifiedNB = ~strcmp(predictionNB,species);
misclassifiedNB = ~(predictionNB == num);
disp([char('Misclassified observations with Naive Bayes: '),num2str((sum(misclassifiedNB) / N)*100),char('%.')])
[confusionMatrixNB,orderNB] = confusionmat(num,predictionNB)
figure, plotconfusion(num', predictionNB'), title('NB confusion matrix')
%
% Plot NB with PCA data
figure
NB = fitcnb(mappedX,num);
gscatter(mappedX(:,1),mappedX(:,2),num);
h = gca;
cxlim = h.XLim;
cylim = h.YLim;
hold on
Params = cell2mat(NB.DistributionParameters); 
Mu = Params(2*(1:2)-1,1:2); % Extract the means
Sigma = zeros(2,2,2);
for j = 1:2
    Sigma(:,:,j) = diag(Params(2*j,:)).^2; % Create diagonal covariance matrix
    xlim = Mu(j,1) + 4*[-1 1]*sqrt(Sigma(1,1,j));
    ylim = Mu(j,2) + 4*[-1 1]*sqrt(Sigma(2,2,j));
    f = @(x,y) arrayfun(@(x0,y0) mvnpdf([x0 y0],Mu(j,:),Sigma(:,:,j)),x,y);
    fcontour(f,[xlim ylim]) % Draw contours for the multivariate normal distributions 
end
h.XLim = cxlim;
h.YLim = cylim;
title('Naive Bayes Classifier -- PCA projected data')
%xlabel('Petal Length (cm)')
%ylabel('Petal Width (cm)')
legend('No disease','Disease')
hold off
%
disp('------------------------------------------------------------')

%NB with kernel distribution

NB = fitcnb(Data,num,'Distribution','kernel');
predictionNB1 = NB.predict(Data); 
%misclassifiedNB = ~strcmp(predictionNB,species);
misclassifiedNB1 = ~(predictionNB1 == num);
disp([char('Misclassified observations with naive bayes classifier with kernel distribution: '),num2str((sum(misclassifiedNB1) / N)*100),char('%.')])
[confusionMatrixNB1,orderNB1] = confusionmat(num,predictionNB1)
figure, plotconfusion(num', predictionNB1'), title('NB with kernel distribution confusion matrix')
disp('------------------------------------------------------------')

%LDA classifier
LDAclassifier = fitcdiscr(Data,num)
predictedClassLabels = resubPredict(LDAclassifier);
misclassifiedLDA = sum(predictedClassLabels~=num)
disp([char('Misclassified observations with LDA: '),num2str((misclassifiedLDA / N)*100),char('%.')])
[confusionMatrixLDA,orderLDA] = confusionmat(num,predictedClassLabels)
figure, plotconfusion(num', predictedClassLabels'), title('LDA confusion matrix')
%
a = -1:0.1:1;
[a1,a2] = meshgrid(a,a);
gridLength = length(a);
x1 = reshape(a1,gridLength*gridLength,1);
x2 = reshape(a2,gridLength*gridLength,1);
LDAclassifier = fitcdiscr(mappedX,num);
gridLabels = predict(LDAclassifier,[x1,x2]);
Colors = lines(length(unique(num)));
figure, gscatter(x1,x2,gridLabels,Colors), title('Separating hyperplane with PCA projected data'),...
    legend('No disease','Disease')
markerstr = 'xd';
hold on, gscatter(mappedX(:,1),mappedX(:,2),num,'kkk',markerstr,8.8)%, legend('No disease','Disease')
disp('------------------------------------------------------------')
%
%asd = Data(1,:)
%predict(LDAclassifier, asd)
%figure, bar(
%a = max(X(:,3));
%b = min(X(:,3));
%a = ((-0.666667+1)/2) * (a-b) + b
%find( SumMissClassified == min(SumMissClassified))
%


%-0.452493*std(X(:,3))+mean(X(:,3))
%0.124824*std(X(:,1))+mean(X(:,1))


%figure, plot(mappedX)

