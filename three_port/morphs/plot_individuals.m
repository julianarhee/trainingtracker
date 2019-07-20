

%% Set file paths:

clear all;
clc


matdir = '/share/coxlab-behavior/mworks-data/three_port_morphs/pnas/matfiles/';

figdir = '/share/coxlab-behavior/mworks-data/three_port_morphs/pnas/figures/';


%% LOAD .mat

plot_curves = 0;

load([matdir, 'P_choice.mat']);

animals = fields(mdata);

D = struct();

for a=1:length(animals)
    curr_animal = animals(a);
    display(curr_animal)
    
    curr_animal = curr_animal{1};
    animal_name = strsplit(curr_animal, '_');
    animal_name = animal_name{1};
    
    data = mdata.(curr_animal);
    nmorphs = size(data,1);
    

    data(:,1) = (data(:,1)/(nmorphs-1))*100; % turn morph # into percent 
    D.(animal_name) = struct;
    D.(animal_name).data = data; %{data};
    D.(animal_name).yaxis = 'choiceR'; %Dlabels = {'choiceR'};


    %% GET FIT

    options.expType        = 'YesNo';

    options.sigmoidName = 'norm';
    % options.sigmoidName = 'logn'; % doesn't work
    % options.sigmoidName = 'weibull'; % doesn't work
    % options.sigmoidName = 'gumbel';

    %     result = psignifit(data,options);
    result = psignifit(D.(animal_name).data, options);
%     
%     % PLOT OPTIONS:
% 
%     plotOptions.h              = gca;                  % axes handle to plot in
%     plotOptions.dataColor      = [0,105/255,170/255];  % colour of the data points
%     plotOptions.plotData       = 1;                    % Shall the data be plotted at all?
%     plotOptions.lineColor      = [0,0,0];              % Colour of the psychometric function
%     plotOptions.lineWidth      = 2;                    % Thickness of the psychometric function
%     plotOptions.xLabel         = 'Morph Percent';     % X-Axis label
%     % plotOptions.yLabel         = 'Percent Choose Right';    % Y-Axis label
%     plotOptions.labelSize      = 15;                   % Font size for labels
%     plotOptions.fontSize       = 10;                   % Tick Font size
%     plotOptions.fontName       = 'Helvetica';          % Font type
%     plotOptions.tufteAxis      = false;                % use custom drawn axis 
%     plotOptions.plotPar        = true;                 % plot indications of threshold and asymptotes
%     plotOptions.aspectRatio    = false;                % sets the aspect ratio to a golden ratio
%     plotOptions.extrapolLength = .2;                   % how far to extrapolate from the data
%                                                        % (in proportion of the data range) 
%     plotOptions.CIthresh       = true;                % plot a confidence interval at threshold
% 
%     
%     %for d=1:length(D)
%     plotOptions.yLabel         = 'Percent Choose Right';    % Y-Axis label
%                 
    %result = psignifit(D{d},options);
%     result = psignifit(D.(animal_name).data, options);
%     
%     if plot_curves
%         plotPsych(result, plotOptions)
% 
%         title(sprintf('%s, fit: %s', animal_name, options.sigmoidName))
% 
%         anchor = Dlabels{d};
%         imname = sprintf('%s_fit_%s_psignifit_pcorrect%s', animal_name, options.sigmoidName, anchor);
%         impath = [figdir, imname]
% 
%         savefig(impath)
%         %saveas(gcf, impath, 'epsc')
%         saveas(gcf, impath, 'png')
%     end
        
    %end
    D.(animal_name).result = result;
    
    D.(animal_name).slope50 = getSlopePC(D.(animal_name).result, 0.5, 1);
    D.(animal_name).thresh50 = getThreshold(D.(animal_name).result, 0.5, 1);

end


%% get curves w/ datapoints and fits for each rat:

D_names = fieldnames(D);
summed_success = [];
summed_total = [];

n_animals = length(D_names)
nrows = 2;
ncols = (n_animals-2)/nrows;

pidx = 1
for animal_idx=1:length(D_names)
    subplot(nrows, ncols, pidx)
    
    animal = D_names{animal_idx}
    if any(strfind(animal,'AG3')) || any(strfind(animal, 'AG11'))
        continue
    end
    display(animal)
    
    summed_success = [summed_success D.(animal).data(:,2)];
    summed_total = [summed_total D.(animal).data(:,3)];
    
    % PLOT OPTIONS:
    plotOptions.h              = gca;                  % axes handle to plot in
    plotOptions.dataColor      = [0,105/255,170/255];  % colour of the data points
    plotOptions.plotData       = 1;                    % Shall the data be plotted at all?
    plotOptions.lineColor      = [.5,.5,.5];              % Colour of the psychometric function
    plotOptions.lineWidth      = 2;                    % Thickness of the psychometric function
    plotOptions.xLabel         = 'Morph level';     % X-Axis label
    % plotOptions.yLabel         = 'Percent Choose Right';    % Y-Axis label
    plotOptions.labelSize      = 12;                   % Font size for labels
    plotOptions.fontSize       = 10;                   % Tick Font size
    plotOptions.fontName       = 'Helvetica';          % Font type
    plotOptions.tufteAxis      = false;                % use custom drawn axis 
    plotOptions.plotPar        = false; %true;                 % plot indications of threshold and asymptotes
    plotOptions.aspectRatio    = true;                % sets the aspect ratio to a golden ratio
    plotOptions.extrapolLength = 0; %.2;                   % how far to extrapolate from the data
                                                       % (in proportion of the data range) 
    plotOptions.CIthresh       = false; %true;                % plot a confidence interval at threshold

    plotOptions.yLabel         = '% choose right';    % Y-Axis label
           
    plotPsych(D.(animal).result, plotOptions)
    hold on;
    frac = D.(animal).data(:,2) ./ D.(animal).data(:,3);
    plot(D.(animal).data(:, 1), frac, '.', 'markersize', 10)
    title(animal)
    
    pidx = pidx + 1;
 
end

%%

pre = 'AG10'; %'AG6';
post = 'AG2';

pre_color = [0, 0, 0];
post_color = [1, 0, 0];

figure();
% PLOT OPTIONS:
plotOptions.h              = gca;                  % axes handle to plot in
plotOptions.dataColor      = [0,105/255,170/255];  % colour of the data points
plotOptions.plotData       = 0;                    % Shall the data be plotted at all?
plotOptions.lineColor      =  pre_color;              % Colour of the psychometric function
plotOptions.lineWidth      = 2;                    % Thickness of the psychometric function
plotOptions.xLabel         = 'Morph level';     % X-Axis label
plotOptions.labelSize      = 12;                   % Font size for labels
plotOptions.fontSize       = 10;                   % Tick Font size
plotOptions.fontName       = 'Helvetica';          % Font type
plotOptions.tufteAxis      = false;                % use custom drawn axis 
plotOptions.plotPar        = false; %true;                 % plot indications of threshold and asymptotes
plotOptions.aspectRatio    = true;                % sets the aspect ratio to a golden ratio
plotOptions.extrapolLength = 0; %.2;                   % how far to extrapolate from the data % (in proportion of the data range) 
plotOptions.CIthresh       = false; %true;                % plot a confidence interval at threshold
plotOptions.yLabel         = '% choose right';    % Y-Axis label


% plot 'pre':
plotOptions.lineColor      = pre_color;              % Colour of the psychometric function
plotPsych(D.(pre).result, plotOptions)
hold on;
plotOptions.lineColor      = post_color;              % Colour of the psychometric function
plotPsych(D.(post).result, plotOptions)

frac = D.(pre).data(:,2) ./ D.(pre).data(:,3);
l1 = plot(D.(pre).data(:, 1), frac, '.', 'DisplayName', 'pre','markersize', 20, 'color', pre_color)

frac = D.(post).data(:,2) ./ D.(post).data(:,3);
l2 = plot(D.(post).data(:, 1), frac, '.', 'DisplayName','post', 'markersize', 20, 'color', post_color)
hold off

legend([l1 l2], {'naive','trained'})

%%

pre = 'AG10'; %'AG6';
post = 'AG2';

pre_color = [0, 0, 0];
post_color = [1, 0, 0];

figure();
h1 = subplot(1,2,1)
% PLOT OPTIONS:
plotOptions.h              = h1;                  % axes handle to plot in
plotOptions.plotData       = 0;                    % Shall the data be plotted at all?
plotOptions.lineColor      =  pre_color;              % Colour of the psychometric function
plotOptions.lineWidth      = 2;                    % Thickness of the psychometric function
plotOptions.xLabel         = 'Morph level';     % X-Axis label
plotOptions.labelSize      = 12;                   % Font size for labels
plotOptions.fontSize       = 10;                   % Tick Font size
plotOptions.fontName       = 'Helvetica';          % Font type
plotOptions.tufteAxis      = false;                % use custom drawn axis 
plotOptions.plotPar        = false; %true;                 % plot indications of threshold and asymptotes
plotOptions.aspectRatio    = true;                % sets the aspect ratio to a golden ratio
plotOptions.extrapolLength = 0; %.2;                   % how far to extrapolate from the data % (in proportion of the data range) 
plotOptions.CIthresh       = false; %true;                % plot a confidence interval at threshold
plotOptions.yLabel         = '% choose right';    % Y-Axis label

% plot 'pre':
plotOptions.lineColor      = pre_color;              % Colour of the psychometric function
plotPsych(D.(pre).result, plotOptions)
hold on;
frac = D.(pre).data(:,2) ./ D.(pre).data(:,3);
l1 = plot(D.(pre).data(:, 1), frac, '.', 'DisplayName', 'pre','markersize', 20, 'color', pre_color)
title('trained')

hold on

h2 = subplot(1,2,2)
% PLOT OPTIONS:
plotOptions.h              = h2;                  % axes handle to plot in
plotOptions.plotData       = 0;                    % Shall the data be plotted at all?
plotOptions.lineColor      =  pre_color;              % Colour of the psychometric function
plotOptions.lineWidth      = 2;                    % Thickness of the psychometric function
plotOptions.xLabel         = 'Morph level';     % X-Axis label
plotOptions.labelSize      = 12;                   % Font size for labels
plotOptions.fontSize       = 10;                   % Tick Font size
plotOptions.fontName       = 'Helvetica';          % Font type
plotOptions.tufteAxis      = false;                % use custom drawn axis 
plotOptions.plotPar        = false; %true;                 % plot indications of threshold and asymptotes
plotOptions.aspectRatio    = true;                % sets the aspect ratio to a golden ratio
plotOptions.extrapolLength = 0; %.2;                   % how far to extrapolate from the data % (in proportion of the data range) 
plotOptions.CIthresh       = false; %true;                % plot a confidence interval at threshold
plotOptions.yLabel         = '% choose right';    % Y-Axis label

plotOptions.lineColor      = post_color;              % Colour of the psychometric function
plotPsych(D.(post).result, plotOptions)
hold on
frac = D.(post).data(:,2) ./ D.(post).data(:,3);
l2 = plot(D.(post).data(:, 1), frac, '.', 'DisplayName','post', 'markersize', 20, 'color', post_color)
title('well trained')

%legend([l1 l2], {'naive','trained'})

