
function import_csv(loadName)
    results_IO = open(loadName)
    D = Dict{String,Vector{Float64}}()
    headingSet = Vector{String}()
    for (i, line) in enumerate(eachline(results_IO))
        splitLine = strip.(split(line, ","))
        if i==1
            headingSet = splitLine
            for s in headingSet
                D[s] = Vector{Float64}()
            end
        else
            for i in eachindex(headingSet)
                push!(D[headingSet[i]], parse(Float64, splitLine[i]))
            end
        end
    end
    close(results_IO)
    return D
end
##

materialOptimisationType = :NeoHookean #:NeoHookean #:Ogden

##

fig = Figure(size=(1200, 800))
ax1 = Axis(fig[1, 1], xlabel = "Displacement (mm)", ylabel = "Force (N)", xticks = 0:2:18, yticks = 0:50:400 )

# colorSet = (:red, :red, :red, :red)
# handleSet = []
#for (indexColor, material_selection) in enumerate((:ECO10, :ECO30, :ECO50, :DS10))
for (material_selection) in (:ECO10, :ECO30, :ECO50, :DS10)

    if  materialOptimisationType == :NeoHookean
        if material_selection == :ECO10
            global loadName_fea = joinpath(loadNamePath, "results", "iFEA_eco10_neo.csv")
            global loadName_exp = joinpath(loadNamePath, "results", "iFEA_eco10_exp.csv")
        elseif material_selection == :ECO30
            global loadName_fea = joinpath(loadNamePath, "results", "iFEA_eco30_neo.csv")
            global loadName_exp = joinpath(loadNamePath, "results", "iFEA_eco30_exp.csv")
        elseif material_selection == :ECO50
            global loadName_fea = joinpath(loadNamePath, "results", "iFEA_eco50_neo.csv")
            global loadName_exp = joinpath(loadNamePath, "results", "iFEA_eco50_exp.csv")
        elseif material_selection == :DS10
            global loadName_fea = joinpath(loadNamePath, "results", "iFEA_DS10_neo.csv")
            global loadName_exp = joinpath(loadNamePath, "results", "iFEA_DS10_exp.csv")
        end
    elseif materialOptimisationType == :Ogden
        if material_selection == :ECO10
            global loadName_fea = joinpath(loadNamePath, "results", "iFEA_eco10_ogden.csv")
            global loadName_exp = joinpath(loadNamePath, "results", "iFEA_eco10_exp.csv")
        elseif material_selection == :ECO30
            global loadName_fea = joinpath(loadNamePath, "results", "iFEA_eco30_ogden.csv")
            global loadName_exp = joinpath(loadNamePath, "results", "iFEA_eco30_exp.csv")
        elseif material_selection == :ECO50
            global loadName_fea = joinpath(loadNamePath, "results", "iFEA_eco50_ogden.csv")
            global loadName_exp = joinpath(loadNamePath, "results", "iFEA_eco50_exp.csv")
        elseif material_selection == :DS10
            global loadName_fea = joinpath(loadNamePath, "results", "iFEA_DS10_ogden.csv")
            global loadName_exp = joinpath(loadNamePath, "results", "iFEA_DS10_exp.csv")
        end
    end
    fit_data_fea = import_csv(loadName_fea)
    U_fea = fit_data_fea["Displacement"]
    F_fea = fit_data_fea["Force"]

    fit_data_exp = import_csv(loadName_exp)
    U_exp = fit_data_exp["Displacement"]
    F_exp = fit_data_exp["Force"]

    hlp1 = lines!(ax1, U_exp, F_exp, color=:black, linewidth=6)
    hlp2 = lines!(ax1, U_fea, F_fea, color=:red, linewidth=4)

    text!(ax1, U_fea[end], F_fea[end], 
    text = string(material_selection),   
    align = (:right, :center),
    offset = (-10, 10),                      
    fontsize = 18,
    color = :black)
    # hlp2 = lines!(ax1, U_fea, F_fea, color=colorSet[indexColor], linewidth=2)

    # push!(handleSet, hlp2)
end

# pushfirst!(handleSet, hlp1)
# Legend(fig[1, 2], handleSet, ["Exp.", "ECO10", "ECO30", "ECO50", "DS10"])

limits!(ax1, 0, 18, 0, 400)
axislegend(ax1, [hlp1, hlp2], ["Exp.", "FEA"], position = :lt,  margin = (20, 20, 20, 20))

screen = display(GLMakie.Screen(), fig)
GLMakie.set_title!(screen, "FEBio example")
