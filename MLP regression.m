%% 
clear;
close all;
clc;

%% Definicja systemu Hammersteina i generacja pomiarów wejścia-wyjścia
N = 2500;
Xn = 2*rand(1, N)-1;
Zn = normrnd(0,1,[1,N])*0.1;
Yn = Hammerstein(Xn, Zn);

%% Wykres zależności wejścia wyjścia systemu
x = -10:0.01:10;
z = zeros(1, length(x));
y = Hammerstein(x, z);
plot(x, y);

%% Dane treningowe oraz testowe
X_train = Xn';
Y_train = Yn';

X_test = -1:0.001:1;
Y_test = Hammerstein(X_test, zeros(1, length(X_test)));

X_test = X_test';
Y_test = Y_test';

[~,~,X_train_std] = Standaryzuj(X_train);
Y_train_std = Y_train;
[~,~,X_test_std] = Standaryzuj(X_test);
Y_test_std = Y_test;

%% Definicja sieci https://www.kdnuggets.com/2019/11/designing-neural-networks.html

numResponses = size(Y_train_std,2);
featureDimension = size(X_train_std,2);

layers = [...
    sequenceInputLayer(featureDimension, 'Name', 'Input layer')
    fullyConnectedLayer(50, 'Name', '1')
    reluLayer('Name', 'a1')
    fullyConnectedLayer(30, 'Name', '2')
    dropoutLayer('Name', 'Drop')
    reluLayer('Name', 'a2')
    fullyConnectedLayer(10, 'Name', '3')
    reluLayer('Name', 'a3')
    fullyConnectedLayer(30, 'Name', '4')
    reluLayer('Name', 'a4')
    fullyConnectedLayer(50, 'Name', '5')
    reluLayer('Name', 'a5')
    fullyConnectedLayer(numResponses, 'Name', '8')];
lgraph = layerGraph(layers);
dlnet = dlnetwork(lgraph);

%% Pętla treningowa
miniBatchSize = 32;
numEpochs = 100;
numObservations = numel(Y_train_std);
numIterationsPerEpoch = floor(numObservations./miniBatchSize);
learnRate = 0.01;
executionEnvironment = "auto";
plots = "training-progress";

if plots == "training-progress"
    figure
    lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
    ylim([0 inf])
    xlabel("Iteration")
    ylabel("Loss")
    grid on
end

iteration = 0;
start = tic;

for epoch = 1:numEpochs
   idx = randperm(numel(Y_train_std));
   X_train_std = X_train_std(idx);
   Y_train_std = Y_train_std(idx);
   
   for i = 1:numIterationsPerEpoch
       iteration = iteration + 1;
       idx = (i-1) * miniBatchSize+1:i * miniBatchSize;
       X = X_train_std(idx);
       Y = Y_train_std(idx);
       
       dlX = dlarray(X', 'CBT');
       dlY = dlarray(Y', 'CBT');
       
       [gradients, loss] = dlfeval(@modelGradients, dlnet, dlX, dlY);
       updateFcn = @(dlnet, gradients) sgdFunction(dlnet, gradients, learnRate);
       dlnet = dlupdate(updateFcn, dlnet, gradients);
       if plots == "training-progress"
           D = duration(0,0,toc(start),'Format','hh:mm:ss');
           addpoints(lineLossTrain,iteration,double(gather(extractdata(loss))))
           title("Epoch: " + epoch + ", Elapsed: " + string(D))
           drawnow
       end
   end
end

%% Predykcja dla danych ze zbioru testowego

dlX_test = dlarray(X_test_std', 'CBT');
dlY_pred = predict(dlnet,dlX_test);
pred = extractdata(dlY_pred);
rmse = sqrt(mean((pred'-Y_test_std).^2))

figure(1);
plot(X_test, Y_test_std);
hold on;
plot(X_test, pred);

figure(2);
plot(Xn, Yn, '.');
hold on;
plot(X_test, pred);

%% Funkcje zobaczyć kołmogorowa może
function [gradients,loss] = modelGradients(dlnet,dlX,Y)
    dlYPred = forward(dlnet, dlX);
    loss = mean((dlYPred-Y).^2);
    gradients = dlgradient(loss,dlnet.Learnables);
end

function parameter = sgdFunction(parameter, gradient, learnRate)
    parameter = parameter - learnRate .* gradient;
end

function [mu, sig, wek] = Standaryzuj(dane)
    mu = mean(dane);
    sig = std(dane);
    wek = (dane - mu) / sig;
end

function Yn = Hammerstein(Xn, Zn)
    Wn = zeros(1, length(Xn));
    for i=1:1:length(Xn)
        Wn(i) = 2*Xn(i)^3 + Zn(i);
    end
    
%     Wn = [zeros(1,4), Wn];
%     Yn = zeros(1, length(Xn));
%     sys = ones(1, 5);
%     for j=5:1:length(Xn)
%         Yn(j-4) = sum(sys .* Wn(j-4:j)) + Zn(i);
%     end
    Yn = Wn;
end
