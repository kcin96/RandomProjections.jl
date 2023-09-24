using Documenter, RandomProjections

makedocs(sitename="RandomProjections.jl",
        pages = ["Home" => "index.md",
                "Guide" => "guide.md"
                ],
        format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true")
        )

deploydocs(
    repo = "github.com/kcin96/RandomProjections.jl.git"
)
