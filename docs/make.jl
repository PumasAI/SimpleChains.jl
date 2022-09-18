using SimpleChains
using Documenter

DocMeta.setdocmeta!(SimpleChains, :DocTestSetup, :(using SimpleChains); recursive = true)

makedocs(;
  modules = [SimpleChains],
  authors = "Chris Elrod <elrodc@gmail.com> and contributors",
  repo = "https://github.com/PumasAI/SimpleChains.jl/blob/{commit}{path}#{line}",
  sitename = "SimpleChains.jl",
  format = Documenter.HTML(;
    prettyurls = get(ENV, "CI", "false") == "true",
    canonical = "https://PumasAI.github.io/SimpleChains.jl",
    assets = String[],
  ),
  pages = [
    "Home" => "index.md",
    "Examples" => ["examples/smallmlp.md", "examples/mnist.md", "examples/custom_loss_layer.md"],
  ],
)

deploydocs(; repo = "github.com/PumasAI/SimpleChains.jl", devbranch = "main")
