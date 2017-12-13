function phi(x,t,Q,p,A,b)
    t*(1/2*dot(x,Q*x)+dot(p,x))-sum(log.(b-A*x))
end

function phi_no_barr(x,Q,p)
    (1/2*dot(x,Q*x)+dot(p,x))
end

function grad(x,t,Q,p,A,b)
    n_i,n_j=size(A)
    grad_Beta_tot=transpose(sum(A./(b-A*x),1))
    t*(Q*x+p)+grad_Beta_tot
end

function hess(x,t,Q,p,A,b)
    hessian_Beta_tot=transpose(A./(b-A*x))*(A./(b-A*x))
    t*Q+hessian_Beta_tot
end

function dampedNewtonStep(x,f,g,h)
    hessian_x=h(x)
    gradient_x=g(x)
    delta_x=- hessian_x \ gradient_x
    Newton_decrement=sqrt.(abs.(- transpose(gradient_x)*delta_x))[1]
    return x+delta_x/(1+Newton_decrement), Newton_decrement^2/2
end

function dampedNewton(x0,f,g,h,tol)
    x_current=x0
    i=0
    history=[]
    gap=10*tol
    println("Damped Newton")
    while ((gap>tol)&(i<1000)&(gap<10e32))
        push!(history,x_current)
        x_current,gap=dampedNewtonStep(x_current,f,g,h)
        i=i+1
    end
    println("Converged")
    return x_current, history
end

function backtrackSearch(f,grad_f_x,x,delta_x)
    t_step=1
    alpha=0.25
    beta=0.75
    feasible=false
    while ((t_step>10e-32))
        try
            difference=f(x+t_step*delta_x)>f(x)+alpha*t_step*dot(grad_f_x,delta_x)
            break
        catch
            t_step=beta*t_step
        end
    end
    print(t_step)
    while f(x+t_step*delta_x)>f(x)+alpha*t_step*dot(grad_f_x,delta_x)
        t_step=beta*t_step
    end
    return t_step
end

function newtonLStep(x,f,g,h)
    hessian_x=h(x)
    gradient_x=g(x)
    delta_x=-hessian_x \ gradient_x
    Newton_decrement=sqrt.(abs.(transpose(gradient_x)*delta_x))[1]
    t_opt=backtrackSearch(f,gradient_x,x,delta_x)
    return x+t_opt*delta_x, Newton_decrement^2/(2)
end

function newtonLS(x0,f,g,h,tol)
    x_current=x0
    i=0
    println("Newton LS")
    history=[]
    gap=10*tol
    while ((gap>tol)&(i<1000)&(gap<10e32))
        push!(history,x_current)
        x_current,gap=newtonLStep(x_current,f,g,h)
        i=i+1
    end
    println("converged")
    return x_current, history
end

function transformSVMPrimal(tau,X,y)
    n_obs,n_dim=size(X)
    Q=diagm(vcat(ones(n_dim),zeros(n_obs)))
    p=vcat(zeros(n_dim),ones(n_obs)/(n_obs*tau))
    a=hcat(y.*X,eye(n_obs))
    a=vcat(a,hcat(zeros(n_obs,n_dim),eye(n_obs)))
    b=vcat(ones(n_obs),zeros(n_obs))
    return Q,p,-a,-b
end

function transformSVMDual(tau,X,y)
    n_obs,n_dim=size(X)
    Q=(y.*X)*transpose(y.*X)
    p=-ones(n_obs)
    a=vcat(eye(n_obs),-eye(n_obs))
    b=vcat(ones(n_obs)/(n_obs*tau),zeros(n_obs))
    return Q,p,a,b
end

function barr_method(Q,p,A,b,x,mu,tol,t=1,method="DampedNewton")
    current_x=x
    m,n=size(A)
    history=[]
    i=0
    while mu*m/t>tol
        i=i+1
        println(i)
        f(x)=phi(x,t,Q,p,A,b)
        g(x)=grad(x,t,Q,p,A,b)
        h(x)=hess(x,t,Q,p,A,b)
        if method=="DampedNewton"
            current_x,x_hist=dampedNewton(current_x,f,g,h,tol)
        else
            current_x,x_hist=newtonLS(current_x,f,g,h,tol)
        end
        t=mu*t
        history=vcat(history,x_hist)
    end
    return current_x, history
end


using DataFrames
df_iris=readtable("iris.txt")
df_subset=df_iris[(df_iris[:Iris_setosa].=="Iris-versicolor")|(df_iris[:Iris_setosa].=="Iris-virginica"),:]
df_subset[:species]=0
df_subset[(df_subset[:Iris_setosa].=="Iris-versicolor"),:species]=1
df_subset[(df_subset[:Iris_setosa].=="Iris-virginica"),:species]=-1

X=Array(df_subset[[2,3,4]])
y=Array(df_subset[:6])
tau=0.001
X=hcat(X,ones(size(X)[1]))
train_index=sample(1:size(X)[1], Int(size(X)[1]*0.6), replace = false)
test_index=Int.(1:100)
filter!(e->e ∉ train_index,test_index)
X_train=X[train_index,:]
X_test=X[test_index,:]
y_train=y[train_index]
y_test=y[test_index]

test_pred_rate=[]
tau_list=[10e-7,10e-6,10e-5, 10e-4, 10e-3, 5*10e-3,10e-2, 2*10e-2, 3*10e-2, 4*10e-2, 5*10e-2,10e-1,1,10,100]
for tau ∈ tau_list
    Q_primal,p_primal,a_primal,b_primal=transformSVMPrimal(tau,X_train,y_train)
    x_0_primal=vcat(zeros(size(X_train)[2]),ones(size(X_train)[1])*1.5)
    weights_svm,history_primal=barr_method(Q_primal,p_primal,a_primal,b_primal,x_0_primal,2,10e-3,1)
    println(mean(y_train.*(X_train*weights_svm[1:4]).>0))

    prediction_rate=mean(y_test.*(X_test*weights_svm[1:4]).>0)
    println(prediction_rate)
    push!(test_pred_rate,prediction_rate)
end

using Gadfly
white_theme = Theme(
    panel_fill="white",
    default_color="red",
    background_color="white"
)


data_tau=convert(DataFrame,hcat(tau_list,(1-test_pred_rate)*100))
error_plot=plot(data_tau, x=:x1, y=:x2,Geom.line,Scale.x_log10,Guide.xlabel("Tau"),Guide.ylabel("Error rate"),white_theme)
draw(PNG("img/error_vs_tau.png", 10inch, 5inch),error_plot)

function plot_duality_gap(tau,X,y,tol,mu,t_0,problem="Primal",method="DampedNewton")
    Q_primal,p_primal,a_primal,b_primal=transformSVMPrimal(tau,X,y)
    x_0_primal=vcat(zeros(size(X)[2]),ones(size(X)[1])*1.5)
    weights_svm,history_primal=barr_method(Q_primal,p_primal,a_primal,b_primal,x_0_primal,mu,tol,t_0,method)

    Q_dual,p_dual,a_dual,b_dual=transformSVMDual(tau,X,y)
    x_0_dual=ones(size(X)[1])/(2*size(X)[1]*tau)
    x_opt_dual,history_dual=barr_method(Q_dual,p_dual,a_dual,b_dual,x_0_dual,mu,tol,t_0,method)

    phi_no_barr_primal(x)=phi_no_barr(x,Q_primal,p_primal)
    phi_no_barr_dual(x)=-phi_no_barr(x,Q_dual,p_dual)

    primal_value=phi_no_barr_primal.(history_primal)
    dual_value=phi_no_barr_dual.(history_dual)
    if (problem=="dual")
        dual_gap=minimum(primal_value)-dual_value
    else
        dual_gap=primal_value-maximum(dual_value)
    end
    data_plot=convert(DataFrame,hcat(1:size(dual_gap)[1],dual_gap))
    data_plot[:mu]=string(mu)
    data_plot[:problem]=problem
    data_plot[:method]=method
    return data_plot
end
first=true
tau=2*10e-3
data_plt=0
for mu ∈ [2, 15, 50, 100]
    for problem ∈ ["dual","primal"]
        for method ∈ ["DampedNewton","LineSearch"]
            print(i)
            if (first==true)
                first=false
                data_plt=plot_duality_gap(tau,X,y,10e-4,mu,1,problem,method)
            end
            if (first==false)
                data_plt=[data_plt;plot_duality_gap(tau,X,y,10e-4,mu,1,problem,method)]
            end
        end
    end
end

plot(data_plt, x=:x1, y=:x2,color=:mu,Geom.line,Scale.x_log10,Guide.xlabel("Newton step")
,Guide.ylabel("Duality gap"),white_theme)
