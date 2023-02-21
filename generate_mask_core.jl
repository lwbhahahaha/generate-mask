### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 0aa4b37f-f9dd-4c4f-b7d5-a76b3e247231
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using DICOM
	using Plots
	using Images
	using CairoMakie
	using Statistics
	using TiffImages
	using Polynomials
	using ProgressBars
	using Base.Threads
	using ImageFeatures
	using ImageFiltering
	using ImageEdgeDetection
end;

# ╔═╡ 6ba7b7b9-1595-449d-9ff1-b4443675a1a9
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
end;

# ╔═╡ 6543b20d-f178-4033-a4db-c870215e2867
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
end;

# ╔═╡ 646a01a1-02ec-4e27-a210-0addbb5f4bbc
function trim_img(img, cup_r, cup_center_row, cup_center_col, tube_centers; tube_trim_r = 80)
	# mask_value = minimum(img)
	mask_value = Int16(-32768)
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
	# output = output .- mask_value
	return output
end;

# ╔═╡ b90b1bbf-d331-4db6-8650-d784e8933db4
function single_image(imgs)
	println("Started proccessing single images...")
	failure_ct = 0
	ct = size(imgs)[1]
	outputs = []
	Threads.@threads for i = 1 : ct
	# for i = 1 : ct
		path, img = imgs[i]
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
				push!(outputs, (path, output))
			end
		end
	end
	println("Finished processing $ct images. $failure_ct Failed.")
	return outputs
end;

# ╔═╡ 8f7cada9-6f58-44a4-b126-56dc0f4ef88c
function dual_image(imgs_pair)
	println("Started proccessing dual images...")
	failure_ct = 0
	outputs = []
	ct = size(imgs_pair)[1]
	Threads.@threads for pair in imgs_pair
		img1_path, img1, img2_path, img2 = pair
		img1_result = locate_cup_center_and_radius(img1)
		img2_result = locate_cup_center_and_radius(img2)
		if typeof(img1_result) <: String
			println("\tFailed to locate the cup: [$img1_path]")
			failure_ct += 1
		elseif typeof(img2_result) <: String
			println("\tFailed to locate the cup: [$img2_path]")
			failure_ct += 1
		else
			img1_center, img1_r, img1_filtered, img1_edges = img1_result
			img1_r *= 0.95
			img2_center, img2_r, img2_filtered, img2_edges = img2_result
			img2_r *= 0.95

			img_big = img1
			r_big, r_small = img1_r, img2_r
			center_big = img1_center
			path_big = img1_path
			if img2_r > img1_r
				img_big = img2
				r_big, r_small = img2_r, img1_r
				center_big = img2_center
				path_big = img2_path
			end
			
			center_row, center_col = center_big[1], center_big[2]
			centers, tube_rs, debug = locate_tubes(img_big, r_big, center_row, center_col)
			if size(centers)[1] == 0
				println("\tFailed to locate tubes: [$path_big]")
				failure_ct += 1
			else
				output = trim_img(img_big, r_small, center_row, center_col, centers; tube_trim_r = maximum(tube_rs) * 1.25)
				push!(outputs, (path_big, output))
			end
		end
	end
	println("Finished processing $ct pair(s). $failure_ct Failed.")
	return outputs
end;

# ╔═╡ 3283b207-cbbc-4a29-9c4b-878e936b9784
function read_single_images()
	println("Loading single images...")
	# read input folder
	imgs = []
	Tiff_ct, DICOM_ct = 0,0
	paths = readdir("input/", join = true)
	for path in paths
		if isfile(path) 
			# DICOM
			if endswith(path, ".slice")
				push!(imgs, (path, dcm_parse(path)[(0x7fe0, 0x0010)]))
				DICOM_ct += 1
			end
			# TIFF
			if endswith(path, ".tif")
				mat = convert(Array{Float64}, Gray.(TiffImages.load(path)))
				mat = mat .* 65535 .- 32768
				mat = Int16.(mat)
				push!(imgs, (path, mat))
				Tiff_ct += 1
			end
		end
	end
	println("\tFound $DICOM_ct DICOM image(s) and $Tiff_ct TIFF image(s).")
	return imgs
end;

# ╔═╡ fa0d19e6-8c70-4858-82fb-3307cf450dbf
function read_pair_images()
	println("Loading pair images...")
	imgs_pair = []
	pair_ct = 0
	folder_paths = readdir("input_pair/", join = true)
	for folder_path in folder_paths
		if isdir(folder_path)
			folder_name = basename(folder_path)
			img_paths = readdir(folder_path, join = true)
			curr_pair = []
			for path in img_paths
				if isfile(path) 
					# DICOM
					if endswith(path, ".slice")
						push!(curr_pair, path, dcm_parse(path)[(0x7fe0, 0x0010)])
					end
					# TIFF
					if endswith(path, ".tif")
						mat = convert(Array{Float64}, Gray.(TiffImages.load(path)))
						mat = mat .* 65535 .- 32768
						mat = Int16.(mat)
						push!(curr_pair, path, mat)
					end
				end
			end
			pair_ct += 1
			push!(imgs_pair, curr_pair)
		end
		
	end
	# 
	println("\tFound $pair_ct pair(s).")
	return imgs_pair
end;

# ╔═╡ c55ecae3-f960-41a1-a6ed-fb71416f6511
function write_images(outputs)
	print("Saving $(size(outputs)[1]) images...")
	Threads.@threads for output in outputs
	# for output in outputs
		path, img_data = output
		splited = split(path, "/")
		_splited = split(splited[1], "_")
		_splited[1] = "output"
		splited[1] = join(_splited, "_")
		output_dir = join(splited[1:end-1], "/")
		isdir(output_dir) || mkdir(output_dir)
		# DICOM
		if endswith(path, ".slice")
			dcm_data = dcm_parse(path)
			dcm_data[(0x7fe0, 0x0010)] = img_data
			path = "modified_" * basename(path)
			dcm_write(joinpath([output_dir, path]), dcm_data)
		end
		# TIFF
		if endswith(path, ".tif")
			img_data = Float64.(img_data)
			img_data = img_data .+ 32768
			img_data = img_data ./ 65535
			img_data = convert(Matrix{ColorTypes.Gray{FixedPointNumbers.N0f16}}, img_data)
			out_img = TiffImages.DenseTaggedImage(img_data)
			# deal with tags
			org_img = TiffImages.load(path)
			out_img_ifds = out_img.ifds[1]
			# delete all tags in new
			for tag in out_img_ifds.tags
				tag_number = tag[2][1].tag
				delete!(out_img_ifds, tag_number)
			end
			# copy tags form org
			for tag in org_img.ifds[1].tags
				tag_number = tag[2][1].tag
				tag_data = tag[2][1].data
				if typeof(tag_data) <: SubString{String}
					tag_data = String(tag_data)
				end
				out_img_ifds[tag_number] = tag_data
			end
			path = "modified_" * basename(path)
			TiffImages.save(joinpath([output_dir, path]), out_img)
		end
	end
	println("Done!")
end;

# ╔═╡ 3f400eff-847a-4872-840e-97ac6330d879
# function find_pairs(imgs)
# 	Dict1 = Dict()
# 	for (i, img_tuple) in enumerate(imgs)
# 		path, img_data = img_tuple
# 		splited = split(basename(path),"_")
# 		# println(splited)
# 		for (j, str) in enumerate(splited)
# 			if str == "low" || str == "high"
# 				splited[j] = ""
# 			end
# 		end
# 		splited = join(splited)
# 		arr = get(Dict1, splited, [])
# 		if arr == []
# 			Dict1[splited] = arr
# 		end
# 		push!(arr, i)
# 	end
# 	pair_img_idx = []
# 	pair_ct = 0
# 	for arr in values(Dict1)
# 		if size(arr)[1] == 2
# 			pair_ct += 1
# 			push!(pair_img_idx, arr) # (path1, mat1, path2, mat2)
# 		end
# 	end
# 	println("Found $pair_ct pair(s).")
# 	return pair_img_idx
# end;

# ╔═╡ d6a2d1aa-a822-11ed-2d6a-a9c9fee66fab
function run!(process_mode = 0)
	flush(stdout)
	println("Process Mode = $process_mode\nOutput Mode = $output_mode")
	if process_mode==0
		imgs = read_single_images()
		outputs = single_image(imgs)
		write_images(imgs)
	else
		imgs = read_pair_images()
		outputs = dual_image(imgs)
		write_images(imgs)
	end
end;

# ╔═╡ 42ba2c6f-a550-44ab-a175-adc03946057d
println(
	"\t*** Instruction ***
	
	1. Put images to the `input` folder;
	2. Call `run!()`.

	Process Mode:
	0: single images;
	1: pair images;
	
	Resutls are saved to `output` folder.")

# ╔═╡ Cell order:
# ╠═0aa4b37f-f9dd-4c4f-b7d5-a76b3e247231
# ╟─6ba7b7b9-1595-449d-9ff1-b4443675a1a9
# ╟─6543b20d-f178-4033-a4db-c870215e2867
# ╠═646a01a1-02ec-4e27-a210-0addbb5f4bbc
# ╟─b90b1bbf-d331-4db6-8650-d784e8933db4
# ╟─8f7cada9-6f58-44a4-b126-56dc0f4ef88c
# ╟─3283b207-cbbc-4a29-9c4b-878e936b9784
# ╟─fa0d19e6-8c70-4858-82fb-3307cf450dbf
# ╠═c55ecae3-f960-41a1-a6ed-fb71416f6511
# ╟─3f400eff-847a-4872-840e-97ac6330d879
# ╠═d6a2d1aa-a822-11ed-2d6a-a9c9fee66fab
# ╠═42ba2c6f-a550-44ab-a175-adc03946057d
