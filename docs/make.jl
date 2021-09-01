using SimpleChains
using Documenter

DocMeta.setdocmeta!(SimpleChains, :DocTestSetup, :(using SimpleChains); recursive=true)

makedocs(;
    modules=[SimpleChains],
    authors="Chris Elrod <elrodc@gmail.com> and contributors",
    repo="https://github.com/chriselrod/SimpleChains.jl/blob/{commit}{path}#{line}",
    sitename="SimpleChains.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://chriselrod.github.io/SimpleChains.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/chriselrod/SimpleChains.jl",
    devbranch="main",
)
