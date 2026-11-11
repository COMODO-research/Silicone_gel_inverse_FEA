using Comodo
using Comodo.GeometryBasics
using Comodo.GLMakie
using Comodo.LinearAlgebra
using Comodo.Statistics
using FEBio
using FEBio.XML
using Printf 
using LeastSquaresOptim # Used for inverse FEA optimisation
using GLM, DataFrames # For experimental data shifting

GLMakie.closeall()

######
# Set FEBio exec path or name
const FEBIO_EXEC = "febio4" # FEBio executable

function export2csv(saveName, U, F)
    results_IO = open(saveName, "w")        

    # Create first heading line in CSV file 
    firstLine = """Displacement, Force\n"""

    # Write first line 
    write(results_IO, firstLine)            

    # Write force
    for i in eachindex(U)
        write(results_IO, @sprintf("%f", U[i]) * ", "
                        * @sprintf("%f", F[i]) * "\n")
    end
    close(results_IO)
end

function R_squared(y, yf)
    y_bar = mean(y)
    SSres = sum((y.-yf).^2)
    SStot = sum((y.-y_bar).^2)
    Rsq = 1.0 - (SSres/SStot)
    return Rsq
end

function lineFit(X, y)
    data = DataFrame(X=X, y=y)
    mdl = lm(@formula(y ~ X), data)
    coefs = coef(mdl)
    a = coefs[2] # Slope
    b = coefs[1] # Intersept
    return a, b
end

function fixteeth!(F, B; method=:add)
    if method == :add # Add interior teeth         
        indNodes_B = unique(reduce(vcat, F[B]))        
        for (i,b) in enumerate(B)
            if !b
                B[i] = all([in(j, indNodes_B) for j in F[i]])
            end
        end
    elseif method == :remove # Remove exterior teeth
        indNodes_B = unique(reduce(vcat, F[.!B]))        
        for (i,b) in enumerate(B)
            if b
                B[i] = !all([in(j, indNodes_B) for j in F[i]])
            end
        end    
    end
    return B
end

function bool_touching(F, B)
    indNodes_B = unique(reduce(vcat, F[B]))        
    B_touch = fill(false, length(B))
    for (i, b) in enumerate(B)
        if !b
            for j in F[i]                
                if in(j, indNodes_B) 
                    B_touch[i] = true
                    break
                end                
            end    
        end
    end  
    return B_touch
end


function subtri_local(F, V, C, B)
    E = meshedges(F)
    Eu, indReverse = gunique(E; return_unique=Val(true), return_inverse=Val(true), sort_entries=true)
    con_E2F = con_edge_face(F,Eu,indReverse)

    EB = meshedges(F[B])
    B_boundary_edges = occursonce(EB; sort_entries=true)
    edgeIndicesBoundary = Vector{Int}()
    E_set = Set{LineFace{Int}}([sort(e) for e in EB[B_boundary_edges]])

    B_touch = fill(false, length(F))
    for (i, e) in enumerate(Eu)
        if in(sort(e), E_set)
            push!(edgeIndicesBoundary, i)
            for j in con_E2F[i]
                if B[j] == false
                    B_touch[j] = true
                end
            end
        end
    end
    Eb = Eu[edgeIndicesBoundary]

    # Keep as is
    F1 = Vector{eltype(F)}()
    C1 = Vector{eltype(C)}()
    for (i, f) in enumerate(F)
        if !(B[i] || B_touch[i])
            push!(F1, f)
            push!(C1, C[i])
        end
    end
    F2 = F[B] # Subdevide normally 
    F2, V2 = remove_unused_vertices(F2, V)
    F2, V2 = subtri(F2, V2, 1)
    C2 = repeat(C[B], outer=4)

    F3 = F[B_touch] # Split along shared edge
    C3 = C[B_touch]

    F3n = Vector{eltype(F1)}()
    V3n = simplexcenter(Eb, V)
    C3n = Vector{eltype(C)}()
    E_vec = sort.(Eb)
    m = length(V) + length(V2) 
    for (i_f, f) in enumerate(F3)
        for j in 1:3    
            e = LineFace{Int}(f[j], f[mod1(j+1, 3)])
            i = findfirst(==(sort(e)), E_vec)
            if !isnothing(i)
                V[f[mod1(j+2, 3)]] = 0.2*V3n[i] + 0.8*V[f[mod1(j+2, 3)]]
                push!(F3n, TriangleFace{Int}(f[j],            m+i, f[mod1(j+2, 3)]))
                push!(F3n, TriangleFace{Int}(m+i, f[mod1(j+1, 3)], f[mod1(j+2, 3)]))
                append!(C3n, fill(C3[i_f],2))
            end
        end
    end

    Fn = [F1; [f.+length(V) for f in F2]; F3n]
    Vn = [V; V2; V3n]
    Cn = [C1; C2; C3n]
    Fn, Vn = remove_unused_vertices(Fn, Vn)
    Fn, Vn = mergevertices(Fn, Vn)

    return Fn, Vn, Cn
end

function quaddisc_pillowrefine(F, V, r, n=1)
    if iszero(n)
        return F, V
    elseif isone(n)        
        E = meshedges(F) # Non-unique edges
        Eu, inv_E = gunique(E; return_unique=Val(true), return_inverse=Val(true), sort_entries=true)
        con_E2F = con_edge_face(F, Eu, inv_E)

        Fn = deepcopy(F)
        Vn = deepcopy(V)    
        boundaryFaceIndexSet = Set{Int}()
        for c in con_E2F
            if length(c)==1
                push!(boundaryFaceIndexSet, c[1])
            end
        end        
        for i in boundaryFaceIndexSet
            f = Fn[i]         
            m = length(Vn)
            
            # Create nodes 
            vp_3 = 2/3*Vn[f[3]] + 1/3*Vn[f[4]]
            vp_4 = 1/3*Vn[f[3]] + 2/3*Vn[f[4]]

            v1 = 2/3*Vn[f[1]] + 1/3*Vn[f[2]]
            v1 = r*v1/norm(v1)
            v2 = 1/3*Vn[f[1]] + 2/3*Vn[f[2]]
            v2 = r*v2/norm(v2)
            v3 = 2/3*v2 + 1/3*vp_3
            v4 = 2/3*v1 + 1/3*vp_4
            v5 = 1/2*v4 + 1/2*vp_4
            v6 = 1/2*v3 + 1/2*vp_3
            v7 = 2/3*Vn[f[1]] + 1/3*Vn[f[4]]
            v8 = 2/3*Vn[f[2]] + 1/3*Vn[f[3]]
            append!(Vn, [v1, v2, v3, v4, v5, v6, v7, v8])
            
            # Create faces 
            f1 = QuadFace{Int}( f[1],   m+1,   m+4,   m+7)
            f2 = QuadFace{Int}(  m+1,   m+2,   m+3,   m+4)
            f3 = QuadFace{Int}(  m+2,  f[2],   m+8,   m+3)
            f4 = QuadFace{Int}(  m+7,   m+4,   m+5,  f[4])
            f5 = QuadFace{Int}(  m+4,   m+3,   m+6,   m+5)
            f6 = QuadFace{Int}(  m+3,   m+8,  f[3],   m+6)
            f7 = QuadFace{Int}(  m+5,   m+6,  f[3],  f[4])        
            append!(Fn, [f1, f2, f3, f4, f5, f6, f7])
        end
        deleteat!(Fn, sort(collect(boundaryFaceIndexSet)))
        Fn, Vn = mergevertices(Fn, Vn)
    else
        Fn = deepcopy(F)
        Vn = deepcopy(V)    
        for _ in 1:n
            Fn, Vn = quaddisc_pillowrefine(Fn, Vn, r, 1)
        end        
    end
    return Fn, Vn
end

function import_csv(loadName)
    results_IO = open(loadName)        
    D = Dict{String, Vector{Float64}}()
    headingSet = Vector{String}()
    for (i,line) in enumerate(eachline(results_IO))
        splitLine = strip.(split(line,","))
        if i==1
            headingSet = splitLine
            for s in headingSet
                D[s] = Vector{Float64}()
            end
        else
            if i == 2
                columnEnd = length(headingSet)
            else
                columnEnd = length(headingSet)-1
            end
            for i in 1:columnEnd
                push!(D[headingSet[i]], parse(Float64, splitLine[i]))
            end
        end
    end
    close(results_IO)
    return D
end

material_selection = :DS10

loadNamePath  = joinpath(dirname(@__DIR__), "assets")
# loadNamePath = "/home/roland/Documents/Julia/Env_Silicone_gel_inverse_FEA/Silicone_gel_inverse_FEA/assets" #joinpath(febiojl_dir(), "assets", "data", "data_cylinder_compression_01.csv")

if material_selection == :ECO10
    loadNameData = joinpath(loadNamePath, "ECO10.csv")
elseif material_selection == :ECO30
    loadNameData = joinpath(loadNamePath, "ECO30.csv")
elseif material_selection == :ECO50
    loadNameData = joinpath(loadNamePath, "ECO50.csv")
elseif material_selection == :DS10
    loadNameData = joinpath(loadNamePath, "DS10.csv")
end

fit_data = import_csv(loadNameData)
U_exp = fit_data["Displacement"]
F_exp = fit_data["Force"]

L0 = fit_data["L0"]

## Shift graph so initial is at 0.0, 0.0 
strainFit = 0.001
b_fit = U_exp .<= (strainFit.*L0)
a, b = lineFit(U_exp[b_fit], F_exp[b_fit])
U_exp .-= -b/a
pushfirst!(U_exp, 0.0)
pushfirst!(F_exp, 0.0)

## Control parameters 

# Loading 
# strainApplied = 0.30 # Equivalent linear strain

# FEA control settings
numTimeSteps = 20 # Number of time steps desired
max_refs = 50 # Max reforms
max_ups = 0 # Set to zero to use full-Newton iterations
opt_iter = 8 # Optimum number of iterations
max_retries = 15 # Maximum number of retries
dtmin = (1.0/numTimeSteps)/100.0 # Minimum time step size
dtmax = 1.0/numTimeSteps  # Maximum time step size
symmetric_stiffness = 1 
min_residual = 1e-20

# Geometry

h = L0[1] - 2.0 # Sample height minus disc heights
r = 25.0 # Sample radius 

elementType = "hex20"# "tet10" # "tet4", "hex8", or "hex20"

if elementType == "hex8" || elementType == "hex20"
    
    n = 1 # Number of split iterations of the top surface 
    nh = 13 #ceil(Int, h/pointSpacing) # Use 0 to switch to automatic

    Fs, Vs = quaddisc(r, n; method=:Catmull_Clark)
    
    nEdgeSplit = 1
    
    Fs, Vs = quaddisc_pillowrefine(Fs, Vs, r, nEdgeSplit)
    pointSpacing = pointspacingmin(Fs, Vs)
    
    E, V = extrudefaces(Fs,Vs; extent=h, direction=:both, num_steps=nh)

    F = element2faces(E)
    indBoundaryFaces = boundaryfaceindices(F)
    
    Fb = F[indBoundaryFaces]
    Fb, Vb = remove_unused_vertices(Fb, V)  
            
    Cb = zeros(Int,length(Fb))
    Nb = facenormal(Fb,Vb)
    for (i,n) in enumerate(Nb)
        d = dot(n,[0.0, 0.0, 1.0])
        if d > 0.5
            Cb[i] = 2
        elseif d<-0.5
            Cb[i] = 1
        else 
            Cb[i] = 3
        end
    end

    pillowOutward = true
    nShrink = 2
    if pillowOutward==true && nShrink>0        
        f_scale = (r - (nShrink*pointSpacing))/r #scale fator for shrinking
        V = [v*f_scale for v in V]

        F = element2faces(E)
        indBoundaryFaces = boundaryfaceindices(F)
        
        Fb = F[indBoundaryFaces]
        Fb, Vb = remove_unused_vertices(Fb, V)  

        indBottom = elements2indices(Fb[Cb.==1])
        indTop = elements2indices(Fb[Cb.==2])
        indSide = elements2indices(Fb[Cb.==3])        

        Nb_V = vertexnormal(Fb, Vb)                   
        indTopBottom = [indTop; indBottom]
        indRings = Vector{Int}()
        for i in indSide
            if in(i, indTopBottom)
                push!(indRings, i)
                nb_v = Nb_V[i]
                d_xy = norm([nb_v[1], nb_v[2]])
                fs = 1.0/d_xy
                Nb_V[i] = Vec{3, Float64}(nb_v[1]*fs, nb_v[2]*fs, nb_v[3]/abs(nb_v[3]))
            end
        end
        
        Ep, Vp = extrudefaces(Fb,Vb; extent=nShrink*pointSpacing, direction=:positive, num_steps=nShrink+1, N=Nb_V)
        E = [E; [eltype(E)(e.+length(V)) for e in Ep]]
        V = [V; Vp]
        V,_,indMap = mergevertices(V; pointSpacing = pointSpacing)
        indexmap!(E, indMap)

        F = element2faces(E)
        
        indBoundaryFaces = boundaryfaceindices(F)
        Fb = F[indBoundaryFaces]
        Cb = zeros(Int,length(Fb))
        Nb = facenormal(Fb,V)
        for (i,n) in enumerate(Nb)
            d = dot(n,[0.0, 0.0, 1.0])
            if d > 0.5
                Cb[i] = 2
            elseif d<-0.5
                Cb[i] = 1
            else 
                Cb[i] = 3
            end
        end
    
        # Scale mesh to correct size
        rMax = maximum([norm([v[1],v[2]]) for v in V])
        hMax = maximum([v[3] for v in V])
        fs_h = (h/2.0)/hMax
        fs_r = r/rMax
        V = [Point{3, Float64}(v[1]*fs_r, v[2]*fs_r, v[3]*fs_h) for v in V]

        # Push top and bottom nodes to not be stretched due to pillowing
        indTopBottom = elements2indices(Fb[Cb.!==3])
        m = 1.0 # Blending factor between not pushed and fully pushed
        fr = m*(1.0/f_scale) +  (1.0-m)#(r+(nShrink*pointSpacing))/r
        for i in indTopBottom
            v = V[i]
            rs = norm([v[1], v[2]])                
            if rs<(r-pointSpacing/100.0)
                V[i] = Point{3, Float64}(fr*v[1], fr*v[2], v[3])
            end
        end

        # Push side nodes to not be stretched due to pillowing
        ind_side = elements2indices(Fb[Cb.==3])
        z = [v[3] for v in V[ind_side]] # Get the z coordinates 

        z_uni, indUni, indReverse = gunique(z; return_unique=Val(true), return_index=Val(true), return_inverse=Val(true))   
        ind_sort = sortperm(z_uni)
        z_fix = range(-h/2.0, h/2.0, length(z_uni))
        for (j,i) in enumerate(ind_side)
            v = V[i]
            zNow = z_fix[ind_sort[indReverse[j]]]            
            V[i] = Point{3, Float64}(v[1], v[2], zNow)
        end

        # Bias side nodes in z-direction so mesh is denser near edges
        ind_bias = elements2indices(Fb[Cb.==3])
        z = [v[3] for v in V[ind_bias]] # Get the z coordinates 
        s = sign.(z) # Get the sign
        b = 1.0
        f = 1.0 .- (abs.(1.0 .- abs.(z./(h/2.0)))).^b
        for (j,i) in enumerate(ind_bias)
            v = V[i]
            V[i] = Point{3, Float64}(v[1], v[2], f[j]*(h/2.0)*s[j])
        end
        
        # Smoothing interior while holding on to outside
        F_uni = gunique(F; sort_entries=true)
        constrained_points = elements2indices(Fb)
        nMax = 50        

        α = 0.1
        β = 0.5
        V = smoothmesh_hc(F_uni, V, nMax,α,β; constrained_points = constrained_points)
        λ = 0.5
        V = smoothmesh_laplacian(F_uni,V,nMax,λ; constrained_points = constrained_points)



        # # E, V = subhex(E,V,1) #split 

        # # F = element2faces(E)
        # # indBoundaryFaces = boundaryfaceindices(F)
        # # Fb = F[indBoundaryFaces]
        # # Cb = zeros(Int,length(Fb))
        # # Nb = facenormal(Fb,V)
        # # for (i,n) in enumerate(Nb)
        # #     d = dot(n,[0.0, 0.0, 1.0])
        # #     if d > 0.5
        # #         Cb[i] = 2
        # #     elseif d<-0.5
        # #         Cb[i] = 1
        # #     else 
        # #         Cb[i] = 3
        # #     end
        # # end

    end

    # Convert to hex20 if needed 
    if elementType == "hex20"  
        E, V = hex8_hex20(E, V)
        F = element2faces(E)
        indBoundary = boundaryfaceindices(F)
        Fb = F[indBoundary]
    end

    ## Create face sets to define node sets later 
    Fb_bottom = Fb[Cb.==1]
    Fb_top = Fb[Cb.==2]    
    Fb_s1 = Fb[Cb.==6]
    Fb_s2 = Fb[Cb.==3]

elseif elementType == "tet4" || elementType == "tet10"    
    pointSpacing = 5.0 # More elements so barreling is captured

    Fb, Vb, Cb = tricylinder(r, h, pointSpacing; face_type=:tri)
    # Vb = [Point{3, Float64}(v[1], v[2], v[3]*2.0) for v in Vb]

    function getrimbool(F, C)
        BF = C.==3
        Eb = boundaryedges(F[BF])        
    
        indNodes_B = elements2indices(Eb)
        B = fill(false, length(F))
        for i in findall(BF)
            for j in F[i]                
                if in(j, indNodes_B) 
                    B[i] = true
                    break
                end                
            end  
        end  
        return B
    end


    function bool_grow(F, B)
        indNodes_B = unique(reduce(vcat, F[B]))        
        B_touch = deepcopy(B)
        for (i, b) in enumerate(B)
            if !b
                for j in F[i]                
                    if in(j, indNodes_B) 
                        B_touch[i] = true
                        break
                    end                
                end  
            end  
        end  
        return B_touch
    end

    B = getrimbool(Fb, Cb)
    # B = bool_grow(Fb, B)
    fixteeth!(Fb, B; method=:add)
    Fb, Vb, Cb = subtri_local(Fb, Vb, Cb, B)

    # B = getrimbool(Fb, Cb)    
    # fixteeth!(Fb, B; method=:add)
    # Fb, Vb, Cb = subtri_local(Fb, Vb, Cb, B)

    stringOpt = "paAqYQ"
    if elementType == "tet10"
        element_type = Tet10{Int}
    else
        element_type = Tet4{Int}
    end
    V_regions = [Point{3, Float64}(0.0, 0.0, 0.0)]
    V_holes = Vector{Point{3, Float64}}()
    vol1 = pointSpacing^3 / (6.0*sqrt(2.0))
    region_vol = [vol1]
    E, V, _, Fb, Cb = tetgenmesh(Fb, Vb; facetmarkerlist=Cb, V_regions=V_regions,V_holes=V_holes, region_vol=region_vol, stringOpt=stringOpt, element_type=element_type)

    ## Create face sets to define node sets later 
    Fb_bottom = Fb[Cb.==2]
    Fb_top = Fb[Cb.==1]    
    Fb_s1 = Fb[Cb.==6]
    Fb_s2 = Fb[Cb.==3]
end

GLMakie.closeall()


##

cmap = Makie.Categorical(:Spectral) 

Fbs,Vbs = separate_vertices(Fb,V)
Cbs_V = simplex2vertexdata(Fbs,Cb)
M = GeometryBasics.Mesh(Vbs,Fbs)


fig = Figure(size=(1200, 800))

ax1 = AxisGeom(fig[1, 1], title = "Boundary faces with boundary markers for the hexahedral mesh")
hidedecorations!(ax1)
hidespines!(ax1)
# hp2 = meshplot!(ax1, Fbs, Vbs; strokewidth=3, color=Cbs_V, colormap=cmap)
# hp3 = normalplot(ax1, Fbs, Vbs; type_flag=:face, color=:black,linewidth=3)

hp2 = meshplot!(ax1, Fbs, Vbs; strokewidth=1, color=:white, stroke_depth_shift=-0.005f0)
# hp2 = meshplot!(ax1, Fb, V; strokewidth=0.5, color=(:white, 0.5), transparency=true)
# dirplot(ax1, Vb, Nb_V)

ax2 = AxisGeom(fig[1, 2], title = "Cut view of hexahedral mesh")
hp3 = meshplot!(ax2, Fbs, Vbs; strokewidth=1, color=:white, stroke_depth_shift=-0.005f0)
hidedecorations!(ax2)
hidespines!(ax2)

VE  = simplexcenter(E,V)
XE = [v[1] for v in VE]
X = [v[1] for v in V]
xMax = maximum(X)
xMin = minimum(X)
numSlicerSteps = 3*ceil(Int,(xMax-xMin)/mean(edgelengths(F,V)))

stepRange = range(xMin,xMax,numSlicerSteps)
hSlider = Slider(fig[2, :], range = stepRange, startvalue = mean(stepRange), linewidth=30)


on(hSlider.value) do x 
    B = XE .>= x
    indShow = findall(B)
    if isempty(indShow)
        hp3.visible=false        
    else        
        hp3.visible=true
        Fs = element2faces(E[indShow])
        Fs,Vs = separate_vertices(Fs,V)
        Ms = GeometryBasics.Mesh(Vs,Fs)
        hp3[1] = Ms
    end
end
slidercontrol(hSlider,ax2)

screen = display(GLMakie.Screen(), fig)
save("hex_mesh.png", fig; px_per_unit = 8)


# Defining displacement of the top surface in terms of x, y, and z components
displacement_prescribed = -maximum(U_exp) #-strainApplied*h

######
# Define file names
saveDir = joinpath(febiojl_dir(),"assets","temp") # Main directory to save FEBio input and output files
if !isdir(saveDir)
    mkdir(saveDir)      
end

filename_FEB = joinpath(saveDir,"febioInputFile_01.feb")   # The main FEBio input file
filename_xplt = joinpath(saveDir,"febioInputFile_01.xplt") # The XPLT file for viewing results in FEBioStudio
filename_log = joinpath(saveDir,"febioInputFile_01_LOG.txt") # The log file featuring the full FEBio terminal output stream
filename_disp = "febioInputFile_01_DISP.txt" # A log file for results saved in same directory as .feb file  e.g. nodal displacements
filename_stress = "febioInputFile_01_STRESS.txt"
filename_force = "febioInputFile_01_REACTION_FORCE.txt"

global iterationCount = 1
global Rsq = 0.0 

function objective_FEA(x)
    if materialOptimisationType == :Ogden
        c1 = x[1]
        m1 = x[2]
        k = 1000 * c1
    elseif materialOptimisationType == :NeoHookean
        c1 = x[1]
        m1 = 2.0
        k = 1000 * c1
    end

   
    
    # function objectiveFunction_FEA(P, U_exp, F_exp)


    ######
    # Define febio input file XML
    doc,febio_spec_node = feb_doc_initialize()

    aen(febio_spec_node,"Module"; type = "solid") # Define Module node: <Module type="solid"/>

    control_node = aen(febio_spec_node,"Control") # Define Control node: <Control>
        aen(control_node,"analysis","STATIC")               
        aen(control_node,"time_steps",numTimeSteps)
        aen(control_node,"step_size",1.0/numTimeSteps)
        aen(control_node,"plot_zero_state",1)
        aen(control_node,"plot_range",@sprintf("%.2f, %.2f",0,-1))
        aen(control_node,"plot_level","PLOT_MAJOR_ITRS")
        aen(control_node,"plot_stride",1)
        aen(control_node,"output_level","OUTPUT_MAJOR_ITRS")
        aen(control_node,"adaptor_re_solve",1)

    time_stepper_node = aen(control_node,"time_stepper"; type = "default")
        aen(time_stepper_node,"max_retries",max_retries)
        aen(time_stepper_node,"opt_iter",opt_iter)
        aen(time_stepper_node,"dtmin",dtmin)
        aen(time_stepper_node,"dtmax",dtmax)
        aen(time_stepper_node,"aggressiveness",0)
        aen(time_stepper_node,"cutback",5e-1)
        aen(time_stepper_node,"dtforce",0)

    solver_node = aen(control_node,"solver"; type = "solid")
        aen(solver_node,"symmetric_stiffness",symmetric_stiffness)
        aen(solver_node,"equation_scheme",1)
        aen(solver_node,"equation_order","default")
        aen(solver_node,"optimize_bw",0)
        aen(solver_node,"lstol",9e-1)
        aen(solver_node,"lsmin",1e-2)
        aen(solver_node,"lsiter",5)
        aen(solver_node,"max_refs",max_refs)
        aen(solver_node,"check_zero_diagonal",0)
        aen(solver_node,"zero_diagonal_tol",0)
        aen(solver_node,"force_partition",0)
        aen(solver_node,"reform_each_time_step",1)
        aen(solver_node,"reform_augment",0)
        aen(solver_node,"diverge_reform",1)
        aen(solver_node,"min_residual",min_residual)
        aen(solver_node,"max_residual",0)
        aen(solver_node,"dtol",1e-3)
        aen(solver_node,"etol",1e-2)
        aen(solver_node,"rtol",0)
        aen(solver_node,"rhoi",0)
        aen(solver_node,"alpha",1)
        aen(solver_node,"beta",2.5e-01)
        aen(solver_node,"gamma",5e-01)
        aen(solver_node,"logSolve",0)
        aen(solver_node,"arc_length",0)
        aen(solver_node,"arc_length_scale",0)
    qn_method_node = aen(solver_node,"qn_method"; type = "BFGS")
        aen(qn_method_node,"max_ups",max_ups)
        aen(qn_method_node,"max_buffer_size",0)
        aen(qn_method_node,"cycle_buffer",0)
        aen(qn_method_node,"cmax",0)
        
    Globals_node   = aen(febio_spec_node,"Globals")

    Constants_node = aen(Globals_node,"Constants")
        aen(Constants_node,"R",8.3140000e-06)
        aen(Constants_node,"T",298)
        aen(Constants_node,"F",9.6485000e-05)

    Material_node = aen(febio_spec_node,"Material")

    material_node = aen(Material_node,"material"; id = "1", name="Material1", type="Ogden")
        aen(material_node, "c1",  c1)
        aen(material_node, "m1",  m1)
        # aen(material_node, "c2",  c1)
        # aen(material_node, "m2", -m1)
        aen(material_node, "k", k)

    # material_node = aen(Material_node,"material"; id = "1", name="Material1", type="Ogden unconstrained")
    #     aen(material_node, "c1",  c1)
    #     aen(material_node, "m1",  m1)
    #     aen(material_node, "c2",  c1)
    #     aen(material_node, "m2", -m1)
    #     aen(material_node, "cp", k)

    Mesh_node = aen(febio_spec_node,"Mesh")

    # Nodes
    Nodes_node = aen(Mesh_node,"Nodes"; name="nodeSet_all")
        for (i,v) in enumerate(V)        
            aen(Nodes_node,"node", join([@sprintf("%.16e",x) for x ∈ v],','); id = @sprintf("%i", i))     
        end
        
    # Elements
    Elements_node = aen(Mesh_node,"Elements"; name="Part1", type=elementType)
        for (i,e) in enumerate(E)     
            aen(Elements_node,"elem", join([@sprintf("%i", i) for i ∈ e], ", "); id = @sprintf("%i", i))
        end
        
    # Node sets
    bcPrescribeList_z = "bcPrescribeList_z"
    bcSupportList_z = "bcSupportList_z"
    indNodes_bcPrescribeList_z = elements2indices(Fb_top)
    indNodes_bcSupportList_z = elements2indices(Fb_bottom)
    aen(Mesh_node,"NodeSet",join([@sprintf("%i",x) for x ∈ indNodes_bcPrescribeList_z],','); name=bcPrescribeList_z)        
    aen(Mesh_node,"NodeSet",join([@sprintf("%i",x) for x ∈ indNodes_bcSupportList_z],','); name=bcSupportList_z)
    
    MeshDomains_node = aen(febio_spec_node, "MeshDomains")
        aen(MeshDomains_node,"SolidDomain"; mat = "Material1", name="Part1")

    Boundary_node = aen(febio_spec_node, "Boundary")

    bc_node = aen(Boundary_node,"bc"; name="zero_displacement_z", node_set=bcSupportList_z, type="zero displacement")
        aen(bc_node,"x_dof",1)
        aen(bc_node,"y_dof",1)
        aen(bc_node,"z_dof",1)

    bc_node = aen(Boundary_node,"bc"; name="zero_displacement_x", node_set=bcPrescribeList_z, type="zero displacement")
        aen(bc_node,"x_dof",1)
        aen(bc_node,"y_dof",0)
        aen(bc_node,"z_dof",0)

    bc_node = aen(Boundary_node,"bc"; name="zero_displacement_y", node_set=bcPrescribeList_z, type="zero displacement")
        aen(bc_node,"x_dof",0)
        aen(bc_node,"y_dof",1)
        aen(bc_node,"z_dof",0)

    bc_node = aen(Boundary_node,"bc"; name="prescribed_disp_z", node_set=bcPrescribeList_z, type="prescribed displacement")
        aen(bc_node,"dof","z")
        aen(bc_node,"value",displacement_prescribed; lc=@sprintf("%i",1))
        aen(bc_node,"relative",@sprintf("%i",0))

    LoadData_node = aen(febio_spec_node,"LoadData")

    load_controller_node = aen(LoadData_node,"load_controller"; id="1", name="LC_1", type="loadcurve")
        aen(load_controller_node,"interpolate","LINEAR")
        
    points_node = aen(load_controller_node,"points")
        aen(points_node,"pt",@sprintf("%.2f, %.2f",0,0))
        aen(points_node,"pt",@sprintf("%.2f, %.2f",1,1))

    Output_node = aen(febio_spec_node,"Output")

    plotfile_node = aen(Output_node,"plotfile"; type="febio")
        aen(plotfile_node,"var"; type="displacement")
        aen(plotfile_node,"var"; type="stress")
        aen(plotfile_node,"var"; type="relative volume")
        aen(plotfile_node,"var"; type="reaction forces")
        aen(plotfile_node,"var"; type="contact pressure")
        aen(plotfile_node,"compression",@sprintf("%i",0))

    logfile_node = aen(Output_node,"logfile"; file=filename_log)
        aen(logfile_node,"node_data"; data="ux;uy;uz", delim=",", file=filename_disp)
        aen(logfile_node,"element_data"; data="sz", delim=",", file=filename_stress)
        aen(logfile_node,"node_data"; data="Rx;Ry;Rz", delim=",", file=filename_force)


    #######
    # Write FEB file
    XML.write(filename_FEB, doc)

    #######
    # Run FEBio
    # try
        run_febio(filename_FEB,FEBIO_EXEC)
        
        #######
        # Import results
        DD_disp = read_logfile(joinpath(saveDir,filename_disp))
        DD_stress = read_logfile(joinpath(saveDir,filename_stress))
        DD_force = read_logfile(joinpath(saveDir,filename_force))
        numInc = length(DD_disp)
        incRange = 0:1:numInc-1

        F = element2faces(E)
        indBoundaryFaces = boundaryfaceindices(F)
        Fb = F[indBoundaryFaces]
        # Fbs, Vs = separate_vertices(Fb,V)
        Sz = [d[1] for d in DD_stress[numInc-1].data] 
        Sz_F = repeat(Sz,inner=6)
        Sz_Fb = Sz_F[indBoundaryFaces]
        Sz_V = simplex2vertexdata(Fb, Sz_Fb,V)

        #######
        # Visualization

        # Create time varying vectors
        global UT = fill(V,numInc) 
        global VT = fill(V,numInc)
        global SzT = fill(Sz_V,numInc)
        global UT_mag = fill(zeros(length(V)),numInc)
        global ut_mag_max = zeros(numInc)
        global Sz_max = zeros(numInc)
        global Sz_min = zeros(numInc)
        global Fz_curve = zeros(numInc)
        global Uz_curve = zeros(numInc)
        time_curve = zeros(numInc)
        @inbounds for i in 0:1:numInc-1    
            UT[i+1] = [Point{3,Float64}(u) for u in DD_disp[i].data]
            VT[i+1] += UT[i+1]
            UT_mag[i+1] = norm.(UT[i+1])
            ut_mag_max[i+1] = maximum(UT_mag[i+1]) 

            Sz = [d[1] for d in DD_stress[i].data]
            Sz_F = repeat(Sz,inner=6)
            Sz_Fb = Sz_F[indBoundaryFaces]
            Sz_V = simplex2vertexdata(Fb, Sz_Fb, V)
            SzT[i+1] = Sz_V
            Sz_min[i+1] = minimum(Sz)
            Sz_max[i+1] = maximum(Sz)
            Fz = [d[3] for d in DD_force[i].data]
            Fz_curve[i+1] = sum(Fz[indNodes_bcPrescribeList_z])
            Uz_curve[i+1] = -mean([u[3] for u in DD_disp[i].data[indNodes_bcPrescribeList_z]])
            time_curve[i+1] = DD_force[i].time
        end

        min_p = minp([minp(V) for V in VT])
        max_p = maxp([maxp(V) for V in VT])

        # Evaluate objective 
        U_interp = range(0.0, Uz_curve[end], numTimeSteps) 
        F_fea_interp = lerp(Uz_curve, Fz_curve, U_interp) 
        F_exp_interp = lerp(U_exp,    F_exp,    U_interp) 
        diff_exp = (F_fea_interp .- F_exp_interp) 
        global Rsq = R_squared(F_exp_interp, F_fea_interp)
        println("R squared =  ",Rsq)

        
        V_plot_UF = Vector{Point{2,Float64}}(undef,length(Uz_curve))
        for i in eachindex(Uz_curve)
            V_plot_UF[i] = Point{2,Float64}(Uz_curve[i], Fz_curve[i])        
        end

        if iterationCount == 1
            fig = Figure(size=(1200, 800))
            global stepStart = incRange[end]
            ax1 = AxisGeom(fig[1, 1], title = "Step: $stepStart", limits=(min_p[1], max_p[1], min_p[2], max_p[2], min_p[3], max_p[3]))
            global hp1 = meshplot!(ax1, Fb, VT[end]; strokewidth=2, color=UT_mag[end], transparency=false, colormap = Reverse(:Spectral),colorrange=(0.0, maximum(ut_mag_max)))
            Colorbar(fig[1, 2], hp1.plots[1], label = "Displacement magnitude [mm]") 
            hidedecorations!(ax1)
            hidespines!(ax1)

            ax2 = AxisGeom(fig[1, 3], title = "Step: $stepStart", limits=(min_p[1], max_p[1], min_p[2], max_p[2], min_p[3], max_p[3]))
            global hp2 = meshplot!(ax2, Fb, VT[end]; strokewidth=2, color=SzT[end], transparency=false, colormap = :viridis,colorrange=(minimum(Sz_min), maximum(Sz_max)))
            Colorbar(fig[1, 4], hp2.plots[1], label = "σ [MPa]") 

            ax3 = Axis(fig[1, 5], title = "Step: $stepStart", aspect = AxisAspect(1), xlabel="Displacement [mm]", ylabel="Force [N]")
            hlp1 = lines!(ax3, U_exp, F_exp, color=:black, linewidth=5)
            hlp2 = lines!(ax3, V_plot_UF, color=:red, linewidth=2)
            Legend(fig[1, 6], [hlp1,hlp2],["Exp.", "FEA"])

            global hl = hlp2

            hsp = scatter!(ax3, V_plot_UF[stepStart+1], markersize=15, color=:red)
            println("Final force: ", V_plot_UF[end])
            println("Number of elements: ",length(E))

            global hs = hsp

            global hSlider = Slider(fig[2, :], range = incRange, startvalue = stepStart,linewidth=30)
            on(hSlider.value) do stepIndex 
                hp1[1] = GeometryBasics.Mesh(VT[stepIndex+1],Fb)
                hp1.color = UT_mag[stepIndex+1]

                hp2[1] = GeometryBasics.Mesh(VT[stepIndex+1],Fb)
                hp2.color = SzT[stepIndex+1]
                    
                hsp[1] = Point{2,Float64}(Uz_curve[stepIndex+1], Fz_curve[stepIndex+1])
                ax1.title = "Step: $stepIndex"
                ax2.title = "Step: $stepIndex"
                ax3.title = "Step: $stepIndex"
            end

            slidercontrol(hSlider,ax1)

            screen = display(GLMakie.Screen(), fig)
            GLMakie.set_title!(screen, "FEBio example")
        else
            hSlider.range = incRange # Update to use the new range
            stepIndex = incRange[end]  
            hl[1] = V_plot_UF 
            hs[1] = V_plot_UF[end]

            hp1[1] = GeometryBasics.Mesh(VT[stepIndex+1],Fb)
            hp1.color = UT_mag[stepIndex+1]

            hp2[1] = GeometryBasics.Mesh(VT[stepIndex+1], Fb)
            hp2.color = SzT[stepIndex+1]
        end

        if save_data == true
            export2csv(saveName_fea, Uz_curve, Fz_curve)
            export2csv(saveName_exp,    U_exp,    F_exp)
        end

        # catch
    #     return fill(NaN, numTimeSteps) # return NaNs if error occured
    # end

    


    global iterationCount += 1
    return diff_exp
end
##

runMode = :testing  # :testing or  :optimisation 
global materialOptimisationType = :Ogden #:NeoHookean #:Ogden

if materialOptimisationType == :Ogden
    c1_ini = 0.1829741220074908
    m1_ini = 1.5598672279666972

    # Initial Ogden parameters 
    c1 = c1_ini
    m1 = m1_ini
    x0 = [c1, m1] # Initial guess
   
    # Best values

    # ECO10
    # c1 = 0.028138124434864447
    # m1 = 1.9999532685014816
    # Rsq = 0.9992369424797335

    # EC030
    # c1 = 0.04847352133320246
    # m1 = 1.5326463713722551
    # Rsq = 0.9996281023803054
     
    # EC050
    # c1 = 0.07216149174989266
    # m1 = 1.3121151444036485
    # Rsq = 0.9998195315883609

    # DS10
    # c1 = 0.1829741220074908
    # m1 = 1.5598672279666972
    # Rsq = 0.9995346914075024

elseif materialOptimisationType == :NeoHookean
    c1_ini = 0.02808891403270298  
    c1 = c1_ini
    x0 = [c1 ] # Initial guess
  
    # Best values

    # ECO10
    # c1 = 0.02808891403270298
    # Rsq = 0.9992564967164734

    # ECO30
    # c1 = 0.04896797373117905
    # Rsq = 0.9994504401459033

    # ECO50
    # c1 = 0.0736743709886812
    # Rsq = 0.9996368114219063

    # DS10
    # c1 = 0.1851271436833846
    # Rsq = 0.9993408093031614
end

# Creating file names for saving optimisation results 
if  materialOptimisationType == :NeoHookean
    if material_selection == :ECO10
        global saveName_fea = joinpath(loadNamePath, "results", "iFEA_eco10_neo.csv")
        global saveName_exp = joinpath(loadNamePath, "results", "iFEA_eco10_exp.csv")
    elseif material_selection == :ECO30
        global saveName_fea = joinpath(loadNamePath, "results", "iFEA_eco30_neo.csv")
        global saveName_exp = joinpath(loadNamePath, "results", "iFEA_eco30_exp.csv")
    elseif material_selection == :ECO50
        global saveName_fea = joinpath(loadNamePath, "results", "iFEA_eco50_neo.csv")
        global saveName_exp = joinpath(loadNamePath, "results", "iFEA_eco50_exp.csv")
    elseif material_selection == :DS10
        global saveName_fea = joinpath(loadNamePath, "results", "iFEA_DS10_neo.csv")
        global saveName_exp = joinpath(loadNamePath, "results", "iFEA_DS10_exp.csv")
    end
elseif materialOptimisationType == :Ogden
    if material_selection == :ECO10
        global saveName_fea = joinpath(loadNamePath, "results", "iFEA_eco10_ogden.csv")
        global saveName_exp = joinpath(loadNamePath, "results", "iFEA_eco10_exp.csv")
    elseif material_selection == :ECO30
        global saveName_fea = joinpath(loadNamePath, "results", "iFEA_eco30_ogden.csv")
        global saveName_exp = joinpath(loadNamePath, "results", "iFEA_eco30_exp.csv")
    elseif material_selection == :ECO50
        global saveName_fea = joinpath(loadNamePath, "results", "iFEA_eco50_ogden.csv")
        global saveName_exp = joinpath(loadNamePath, "results", "iFEA_eco50_exp.csv")
    elseif material_selection == :DS10
        global saveName_fea = joinpath(loadNamePath, "results", "iFEA_DS10_ogden.csv")
        global saveName_exp = joinpath(loadNamePath, "results", "iFEA_DS10_exp.csv")
    end
end

if runMode == :optimisation 
    x_tol = 1e-3 # Parameter tolerance 
    f_tol = 1e-3 # Function tolerance
    max_number_iterations = 100

    global save_data = false 
    W = optimize(objective_FEA, x0, LevenbergMarquardt(); x_tol=x_tol, f_tol=f_tol, iterations=max_number_iterations);
    
    global save_data = true
    objective_FEA(W.minimizer);

    println("And the winner is....: ", W.minimizer)
    println("R_squared = ", Rsq)

elseif runMode == :testing
    global save_data = true    
    objective_FEA(x0);
end

##