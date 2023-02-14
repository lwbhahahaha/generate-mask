### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ d6a2d1aa-a822-11ed-2d6a-a9c9fee66fab
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using DICOM
	using Plots
	using Images
	using CairoMakie
	using Statistics
	using Polynomials
	using ProgressBars
	using Base.Threads
	using ImageFeatures
	using ImageFiltering
	using ImageEdgeDetection
	
	function locate_cup_center_and_radius(img)
	    img_filtered = mapwindow(median, img, (5,5))
	    img_filtered = imfilter(img_filtered, Kernel.gaussian(17))
	    img_filtered = imfilter(img_filtered, Kernel.gaussian(13))
	    img_edges = canny(img_filtered, (ImageFeatures.Percentile(99), ImageFeatures.Percentile(0)))
	    dx, dy=imgradients(img_filtered, KernelFactors.ando5)
	    img_phase = phase(dx, dy)
	    centers, radii = hough_circle_gradient(img_edges, img_phase, 450:675)
	    if size(centers)[1] == 1
	        return centers[1], radii[1], img_filtered, img_edges
	    else
	        return "Error"
	    end
	end
	
	function locate_tubes(img, r, center_row, center_col)
	    # size of kernel
	    x = round(-24.2366 + 0.0715 * r)
	    x -= (x+1)%2
	    # filter
	    img_filtered = mapwindow(median, img, (5,5))
	    img_filtered = imfilter(img_filtered, Kernel.gaussian(x))
	    img_filtered = imfilter(img_filtered, Kernel.gaussian(x-2))
	    # detect edge
	    img_edges = canny(img_filtered, (ImageFeatures.Percentile(99), ImageFeatures.Percentile(0)))
	    # cropping
	    r_2 = r * r
	    Threads.@threads for i in CartesianIndices(img_edges)
	        r, c = i[1], i[2]
	        d_2 = (r - center_row)^2 + (c - center_col)^2
	        if d_2 > r_2 || d_2 <= 10000
	            img_edges[i] = 0
	        end
	    end
	    # detect circles
	    dx, dy=imgradients(img_filtered, KernelFactors.ando5)
	    img_phase = phase(dx, dy)
	    centers, radii = hough_circle_gradient(img_edges, img_phase, 40:60)
	    return centers, radii, img_edges
	end
	
	function trim_img(img, cup_r, cup_center_row, cup_center_col, tube_centers; tube_trim_r = 80)
	    mask_value = minimum(img)
	    output = deepcopy(img)
	    cup_r_2 = cup_r * cup_r
	    tube_r_2 = tube_trim_r * tube_trim_r
	    Threads.@threads for i in CartesianIndices(output)
	        r, c = i[1], i[2]
	        if ((r - cup_center_row)^2 + (c - cup_center_col)^2) > cup_r_2
	            # trim background
	            output[i] = mask_value
	        else
	            # trim tubes
	            for center in tube_centers 
	                if ((r - center[1])^2 + (c - center[2])^2) <= tube_r_2
	                    output[i] = mask_value
	                end
	            end
	        end
	    end    
	    # rearrage all pixel values
	    output = output .- mask_value
	    return output
	end
	
	function run!()
	    println("Started...")
	    # read input folder
	    imgs = []
	    paths = readdir("input/", join = true)
	    for path in paths
	        if isfile(path) && endswith(path, ".slice")
	            push!(imgs, (path, dcm_parse(path)))
	        end
	    end
	    println("\tFound $(size(imgs)[1]) images...")
	    failure_ct = 0
	    ct = size(imgs)[1]
	    Threads.@threads for i = ProgressBar(1 : ct)
	        path, dcm_data = imgs[i]
	        img = dcm_data[(0x7fe0, 0x0010)]
	        result = locate_cup_center_and_radius(img)
	        if typeof(result) <: String
	            println("\tFailed to locate the cup: [$path]")
	            failure_ct += 1
	        else
	            center, r, img_filtered, img_edges = result
	            r = r * 0.9
	            center_row, center_col = center[1], center[2]
	            centers, tube_rs, debug = locate_tubes(img, r, center_row, center_col)
	            if size(centers)[1] == 0
	                println("\tFailed to locate tubes: [$path]")
	                failure_ct += 1
	            else
	                output = trim_img(img, r, center_row, center_col, centers; tube_trim_r = maximum(tube_rs) * 1.25)
	                # Plots.heatmap(output)
	                dcm_data[(0x7fe0, 0x0010)] = output
	                path = "filtered_" * basename(path)
	                dcm_write(joinpath(["output/", path]), dcm_data)
	            end
	        end
	    end
	    println("Finished processing $(size(imgs)[1]) images. $failure_ct Failed.")
	end
	println(
	"\t*** Insturction ***
	
	1. Put images to the `input` folder;
	2. Call `run!()`.
	
	Resutls are saved to `output` folder.")
end

# ╔═╡ Cell order:
# ╠═d6a2d1aa-a822-11ed-2d6a-a9c9fee66fab
