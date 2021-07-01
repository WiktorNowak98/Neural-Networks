%% Praca magisterska
clc;
clear;
close all;

%% Generacja danych treningowych oraz testowych

% Dane treningowe
N = 2500;
Xn = 2*rand(1, N)-1;
Zn = normrnd(0, 1, [1,N]);
b = ones(1, 25)';
Yn = Hammerstein(Xn, Zn, b)';
%Yn = Nielinowy_statyczny(Xn);

% Dane testowe
x = -1:0.01:1;
y  = Nielinowy_statyczny(x)';

[X_train_std, Y_train_std, X_test_std, Y_test_std] = Preprocess_data(Xn, Yn, x, y);

%% Wykres chmury
figure(1);
plot(Xn, Yn, '.');
hold on;
plot(x, y);

%% Definicja sieci
dlnet = MLP(X_train_std, Y_train_std);
%dlnet = GRU(X_train_std, Y_train_std);

%% Trenowanie sieci
miniBatchSize = 128;
numEpochs = 300;
learnRate = 0.001;

dlnet = Train_SGD(dlnet, X_train_std, Y_train_std, miniBatchSize, numEpochs, learnRate);

%% Predykcja
dlX_test = dlarray(X_test_std', 'CBT');
dlY_pred = predict(dlnet,dlX_test);
pred = extractdata(dlY_pred);
rmse = sqrt(mean((pred-Y_test_std).^2))
loss = (pred - Y_test_std).^2;

%% Wykresy
figure(1);
subplot(2,1,1);
plot(x, Y_test_std);
hold on;
plot(x, pred);
title("Porównanie predykcji sieci z nieliniowością systemu Hammersteina");
legend("Dane testowe","Predykcja");
subplot(2,1,2);
stem(x, loss);
title("Punktowy błąd predykcji");

figure(2);
plot(Xn, Yn, '.');
hold on;
plot(x, pred);
hold on;
plot(x, Y_test_std);
title("Porównanie predykcji sieci z nieliniowością systemu Hammersteina na danych treningowych");
legend("Dane treningowe","Predykcja","Dane testowe");

%% Funkcje małe i duże - https://ruder.io/optimizing-gradient-descent/index.html#stochasticgradientdescent
function dlnet = Train_SGD(dlnet, X_train_std, Y_train_std, batch, ep, rate)
    miniBatchSize = batch;
    numEpochs = ep;
    numObservations = numel(Y_train_std);
    numIterationsPerEpoch = floor(numObservations./miniBatchSize);
    learnRate = rate;
    plots = "training-progress";

    if plots == "training-progress"
        figure
        lineLossTrain = animatedline('Color',[0.85 0.325 0.098],'Marker','o');
        ylim([0 inf])
        xlabel("Epoch")
        ylabel("Loss - MSE")
        grid on
        grid minor
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
       end
       if plots == "training-progress"
           D = duration(0,0,toc(start),'Format','hh:mm:ss');
           addpoints(lineLossTrain,epoch,double(gather(extractdata(loss))))
           title("Epoch: " + epoch + ", Elapsed: " + string(D))
           drawnow
       end
    end
end

function dlnet = Train_MOM(dlnet, X_train_std, Y_train_std, batch, ep, rate)
    miniBatchSize = batch;
    numEpochs = ep;
    numObservations = numel(Y_train_std);
    numIterationsPerEpoch = floor(numObservations./miniBatchSize);
    learnRate = rate;
    plots = "training-progress";
    v = 0;
    if plots == "training-progress"
        figure
        lineLossTrain = animatedline('Color',[0.85 0.325 0.098],'Marker','o');
        ylim([0 inf])
        xlabel("Epoch")
        ylabel("Loss - MSE")
        grid on
        grid minor
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
           updateFcn = @(dlnet, gradients) momFunction(dlnet, gradients, learnRate, v);
           [dlnet, v] = dlupdate(updateFcn, dlnet, gradients);
       end
       if plots == "training-progress"
           D = duration(0,0,toc(start),'Format','hh:mm:ss');
           addpoints(lineLossTrain,epoch,double(gather(extractdata(loss))))
           title("Epoch: " + epoch + ", Elapsed: " + string(D))
           drawnow
       end
    end
end

function dlnet = GRU(X_train_std, Y_train_std)
    numResponses = size(Y_train_std,2);
    featureDimension = size(X_train_std,2);
    layers = [...
        sequenceInputLayer(featureDimension, 'Name', 'Input')
        gruLayer(125, 'Name','gru1')
        dropoutLayer(0.2, 'Name', 'Drop1')
        gruLayer(100, 'Name','gru2')
        dropoutLayer(0.2, 'Name', 'Drop2')
        gruLayer(50, 'Name','gru3')
        dropoutLayer(0.2, 'Name', 'Drop3')
        fullyConnectedLayer(50, 'Name', '1')
        fullyConnectedLayer(numResponses, 'Name', '2')];
    lgraph = layerGraph(layers);
    dlnet = dlnetwork(lgraph);
end

function dlnet = MLP(X_train_std, Y_train_std)
    numResponses = size(Y_train_std,2);
    featureDimension = size(X_train_std,2);
    layers = [...
        sequenceInputLayer(featureDimension, 'Name', 'Input layer')
        fullyConnectedLayer(50, 'Name', '1')
        reluLayer('Name', 'a1')
        fullyConnectedLayer(30, 'Name', '2')
        dropoutLayer(0.5, 'Name', 'Drop')
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
end

function [X_train_std, Y_train_std, X_test_std, Y_test_std] = Preprocess_data(Xn, Yn, x, y)
    X_train = Xn';
    Y_train = Yn';
    
    X_test = x';
    Y_test = y';
    
    [~,~,X_train_std] = Standarize(X_train);
    Y_train_std = Y_train;
    [~,~,X_test_std] = Standarize(X_test);
    Y_test_std = Y_test;
end

function [gradients,loss] = modelGradients(dlnet,dlX,Y)
    dlYPred = forward(dlnet, dlX);
    loss = mean((dlYPred-Y).^2); % MSE
    %loss = max(abs(dlYPred-Y)); % Kołmogorow - Smirnow
    gradients = dlgradient(loss, dlnet.Learnables);
end

function [parameter, v] = momFunction(parameter, gradient, learnRate, v)
%     if(isa(v, 'table'))
%         return;
%     end
    gamma = 0.9;
    v = gamma .* v + learnRate .* gradient;
    parameter = parameter - v;
end

function parameter = sgdFunction(parameter, gradient, learnRate)
    parameter = parameter - learnRate .* gradient;
end

function [mu, sig, wek] = Standarize(data)
    mu = mean(data);
    sig = std(data);
    wek = (data - mu) / sig;
end

function Yn = Hammerstein(Un, Zn, b)
    Wn = Nielinowy_statyczny(Un);
    Yn = Liniowy_dynamiczny(Wn', Zn', b);
end

function Wn = Nielinowy_statyczny(Un)
    Wn = atan(3*Un);
end

function Yn = Liniowy_dynamiczny(Wn, Zn, b)
    phi = zeros(1,length(b));
    PhiN = [];
    for i=1:1:length(Wn)
        phi = [Wn(i), phi];
        phi(end) = [];
        PhiN = [PhiN; phi];
    end
    Yn = PhiN * b + Zn;
end