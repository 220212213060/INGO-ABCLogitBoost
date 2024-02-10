function [bestxb,best_so_far] = INGO_ABC(dim,lb,ub,params,fobj)
%% Parameter settings
Search_Agents=10;
Max_iterations=50;
X=[];
X_new=[];
fit=[];
fit_new=[];
NGO_curve=zeros(1,Max_iterations);
% Initialize population and fitness values
Positions1 = rand(Search_Agents,dim);
% Cubic Chaos Imprint Improved NGO Algorithm
for i=1:Search_Agents
    for j = 1:dim-1
        Positions1(i,j+1) = 2.595*Positions1(i,j)*(1-(Positions1(i,j))^2);
    end
end
for i=1:Search_Agents
    for j = 1:dim
        X(i,j)=Positions1(i,j)*(ub(j)-lb(j))+lb(j);
    end
end
for i =1:Search_Agents
    fit(i)=fobj(params,X(i, :));
end
totalIterations = Max_iterations;
h = waitbar(0, 'Calculations in progress...');
% Algorithm Iteration
for t=1:Max_iterations
    waitbar(t / totalIterations, h, sprintf('Calculations in progress... %d/%d', t, totalIterations));
    [best , blocation]=min(fit);
    if t==1
        xbest=X(blocation,:);
        fbest=best;
    elseif best<fbest
        fbest=best;
        xbest=X(blocation,:);
    end
    %% UPDATE Northern goshawks based on PHASE1 and PHASE2
    for i=1:Search_Agents
        %% Phase 1: Exploration
        I=round(1+rand);
        k=randperm(Search_Agents,1);
        P=X(k,:);
        F_P=fit(k);
        if fit(i)> F_P
            X_new(i,:)=X(i,:)+rand(1,dim) .* (P-I.*X(i,:));
        else
            X_new(i,:)=X(i,:)+rand(1,dim) .* (X(i,:)-P);
        end
        X_new(i,:) = max(X_new(i,:),lb);X_new(i,:) = min(X_new(i,:),ub);
        fit_new(i)=fobj(params,X_new(i, :));
        if(fit_new(i)<fit(i))
            X(i,:) = X_new(i,:);
            fit(i) = fit_new(i);
        end
        %% END PHASE 1
        %% PHASE 2 Exploitation
        R=0.02*(1-t/Max_iterations);
        X_new(i,:)= X(i,:)+ (-R+2*R*rand(1,dim)).*X(i,:);
        X_new(i,:) = max(X_new(i,:),lb);X_new(i,:) = min(X_new(i,:),ub);
        fit_new(i)=fobj(params,X_new(i,:));
        if(fit_new(i)<fit(i))
            X(i,:) = X_new(i,:);
            fit(i) = fit_new(i);
        end
        %% END PHASE 2
    end
    %% SAVE BEST SCORE
    best_so_far(t)=fbest;
    average(t) = mean (fit);
    Score=fbest;
    Best_pos=xbest;
    NGO_curve(t)=Score;
end
bestxb = [floor(Best_pos(1: 2)), Best_pos(3)];
end

