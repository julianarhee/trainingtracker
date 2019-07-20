

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


%% get group curve:

D_names = fieldnames(D);
summed_success = [];
summed_total = [];

for animal_idx=1:length(D_names)
    
    animal = D_names{animal_idx}
    if strfind(animal,'AG3')
        continue
    end
    display(animal)
    
    summed_success = [summed_success D.(animal).data(:,2)];
    summed_total = [summed_total D.(animal).data(:,3)];
    
    % PLOT OPTIONS:
    plotOptions.h              = gca;                  % axes handle to plot in
    plotOptions.dataColor      = [0,105/255,170/255];  % colour of the data points
    plotOptions.plotData       = 0;                    % Shall the data be plotted at all?
    plotOptions.lineColor      = [.5,.5,.5];              % Colour of the psychometric function
    plotOptions.lineWidth      = 1;                    % Thickness of the psychometric function
    plotOptions.xLabel         = 'Morph Percent';     % X-Axis label
    % plotOptions.yLabel         = 'Percent Choose Right';    % Y-Axis label
    plotOptions.labelSize      = 15;                   % Font size for labels
    plotOptions.fontSize       = 10;                   % Tick Font size
    plotOptions.fontName       = 'Helvetica';          % Font type
    plotOptions.tufteAxis      = false;                % use custom drawn axis 
    plotOptions.plotPar        = false; %true;                 % plot indications of threshold and asymptotes
    plotOptions.aspectRatio    = true;                % sets the aspect ratio to a golden ratio
    plotOptions.extrapolLength = 0; %.2;                   % how far to extrapolate from the data
                                                       % (in proportion of the data range) 
    plotOptions.CIthresh       = false; %true;                % plot a confidence interval at threshold

    plotOptions.yLabel         = 'Percent Choose Right';    % Y-Axis label
           
        
    
    plotPsych(D.(animal).result, plotOptions)
    hold on;
 
end
%% PLOT GROUP:

% xlength   = max(result.data(:,1))-min(result.data(:,1));
% x         = linspace(min(result.data(:,1)),max(result.data(:,1)),1000);
% xLow      = linspace(min(result.data(:,1))-plotOptions.extrapolLength*xlength,min(result.data(:,1)),100);
% xHigh     = linspace(max(result.data(:,1)),max(result.data(:,1))+plotOptions.extrapolLength*xlength,100);
% 
% fitValuesLow    = (1-result.Fit(3)-result.Fit(4))*arrayfun(@(x) result.options.sigmoidHandle(x,result.Fit(1),result.Fit(2)),xLow)+result.Fit(4);
% fitValuesHigh   = (1-result.Fit(3)-result.Fit(4))*arrayfun(@(x) result.options.sigmoidHandle(x,result.Fit(1),result.Fit(2)),xHigh)+result.Fit(4);
% 
% fitValues = (1-result.Fit(3)-result.Fit(4))*arrayfun(@(x) result.options.sigmoidHandle(x,result.Fit(1),result.Fit(2)),x)+result.Fit(4);
% plot(x,     fitValues,          'Color', plotOptions.lineColor,'LineWidth',plotOptions.lineWidth)
% plot(xLow,  fitValuesLow,'--',  'Color', plotOptions.lineColor,'LineWidth',plotOptions.lineWidth)
% plot(xHigh, fitValuesHigh,'--', 'Color', plotOptions.lineColor,'LineWidth',plotOptions.lineWidth)
% 
% if result.options.logspace
%     set(gca,'XScale','log')
% end

group_data = zeros(size(D.AG2.data));
group_data(:, 1) = D.(animal).data(:,1); % X-vals are the same
% group_data(:, 2) = mean(summed_success, 2);
% group_data(:, 3) = mean(summed_total, 2);
means_by_subject = summed_success./summed_total;
group_data(:, 2) = mean(means_by_subject,2)*100;
group_data(:, 3) = ones(size(D.AG2.data(:,1)))*100;

% FIT AVERAGED DATA:
group_result = psignifit(group_data, options);

D.average = struct();
D.average.data = group_data;
D.average.result = group_result;
D.average.slope50 = getSlopePC(D.average.result, 0.5, 1);
D.average.thresh50 = getThreshold(D.average.result, 0.5, 1);

%%
% PLOT OPTIONS:
% plotOptions.h              = gca;                  % axes handle to plot in
% plotOptions.dataColor      = [0,105/255,170/255];  % colour of the data points
% plotOptions.plotData       = 0;                    % Shall the data be plotted at all?
plotOptions.lineColor      = [0,0,0];              % Colour of the psychometric function
plotOptions.lineWidth      = 5;                    % Thickness of the psychometric function
plotOptions.xLabel         = 'Morph Percent';     % X-Axis label
% plotOptions.yLabel         = 'Percent Choose Right';    % Y-Axis label
plotOptions.labelSize      = 15;                   % Font size for labels
plotOptions.fontSize       = 10;                   % Tick Font size
plotOptions.fontName       = 'Helvetica';          % Font type
plotOptions.tufteAxis      = false;                % use custom drawn axis 
plotOptions.plotPar        = true; %true;                 % plot indications of threshold and asymptotes
plotOptions.aspectRatio    = true;                % sets the aspect ratio to a golden ratio
plotOptions.extrapolLength = 0; %.2;                   % how far to extrapolate from the data
                                                   % (in proportion of the data range) 
plotOptions.CIthresh       = false; %true;                % plot a confidence interval at threshold

plotOptions.yLabel         = 'Percent Choose Right';    % Y-Axis label
 
plotPsych(D.average.result, plotOptions)


%% Fix axes:
offsetAxes(gca, 0.01)

curr_xlim = xlim;
curr_ylim = ylim;
xlim([curr_xlim(1), 100]);
ylim([curr_ylim(1), 1]);

xticks = get(gca,'XTick');
xticks(end+1) = 100;
xticklabels = cellstr(get(gca,'XTickLabel'));
xticklabels{end+1} = '100';
[xticks,idx] = sort(xticks);
xticklabels = xticklabels(idx);
set(gca,'Xtick',xticks,'XTickLabel',xticklabels);

yticks = get(gca,'YTick');
yticks(end+1) = 1;
yticklabels = cellstr(get(gca,'YTickLabel'));
yticklabels{end+1} = '1';
[yticks,idx] = sort(yticks);
yticklabels = yticklabels(idx);
set(gca,'Ytick',yticks,'YTickLabel',yticklabels);

%

title(sprintf('Group average, fit: %s', options.sigmoidName))

imname = sprintf('ALL_fit_%s_psignifit', options.sigmoidName);
impath = [figdir, imname]

savefig(impath)
%saveas(gcf, impath, 'epsc')
saveas(gcf, impath, 'png')

%% Get slope/threshold info:

% slope =
% 
%     0.0136
%     
% threshold =
% 
%    31.4103
% 
% 
% CI =
% 
%    21.3080   35.7342
%    22.2413   34.8583
%    24.6788   32.9497

D_names = fieldnames(D); % re-assign D names since now have added average
slopes = [];
thresholds = [];

for animal_idx=1:length(D_names)
    
    animal = D_names{animal_idx};
    if strfind(animal,'AG3')
        continue
    end
    display(animal)
   
    slopes = [slopes D.(animal).slope50];
    thresholds = [thresholds D.(animal).thresh50];
    
end
    
%%

figure()
subplot(121)
plot(ones(1,7), thresholds(1:end-1), '.', 'MarkerSize', 20, 'MarkerFaceColor', 'k');
xlim([0,2])
hold on

averaged_threshold = mean(thresholds(1:end-1));
threshold_sem = std(thresholds(1:end-1))/sqrt(length(thresholds(1:end-1)));
plot(1, averaged_threshold, '.', 'MarkerSize', 20, 'MarkerFaceColor', 'r');
hold on
errorbar(averaged_threshold, threshold_sem, 'k')
% ylim([0, 60])
hold on
set(gca,'TickDir','out')
set(gca,'xtick',[]);
set(gca,'xcolor',[1 1 1])
set(gca,'box','off','color','white')


subplot(122)
plot(ones(1,7), slopes(1:end-1), '.', 'MarkerSize', 20, 'MarkerFaceColor', 'k');
hold on
xlim([0,2])

averaged_slope = mean(slopes(1:end-1));
slope_sem = std(slopes(1:end-1))/sqrt(length(slopes(1:end-1)))
plot(1, averaged_slope, '.', 'MarkerSize', 20, 'MarkerFaceColor', 'r');
hold on
errorbar(averaged_slope, slope_sem, 'k')
ylim([0,0.03])

set(gca,'TickDir','out')
hold on
set(gca,'TickDir','out')
set(gca,'xtick',[]);
set(gca,'xcolor',[1 1 1])
set(gca,'box','off','color','white')


% 
% for animal_idx=1:length(D_names)-1
%     
%     animal = D_names{animal_idxs};
%     if strfind(animal,'AG3')
%         continue
%     end
%     
%     slope_coords = [1, D.(animal).slope50];
%     
%     
%     
%     
%     
% end
































    