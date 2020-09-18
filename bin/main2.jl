#####
##### Testing multithreaded self-play
#####

using Distributed


@everywhere const NUM_GAMES = 500

@everywhere ENV["JULIA_CUDA_MEMORY_POOL"] = "split"

@everywhere ENV["ALPHAZERO_DEFAULT_DL_FRAMEWORK"] = "FLUX"

@everywhere using AlphaZero
@everywhere using ProgressMeter

@everywhere include("../games/connect-four/main.jl")
@everywhere using .ConnectFour: Game, Training

@everywhere struct Handler
  progress :: Progress
  Handler(n) = new(Progress(n))
end
@everywhere AlphaZero.Handlers.game_played(h::Handler) = next!(h.progress)
@everywhere AlphaZero.Handlers.checkpoint_game_played(h::Handler) = next!(h.progress)

@everywhere function bench_self_play(n)
  params, benchmark = Training.params, Training.benchmark
  params = Params(params, self_play=SelfPlayParams(params.self_play, num_games=n))
  network = Training.Network{Game}(Training.netparams)
  env = AlphaZero.Env{Game}(params, network)
  return @timed AlphaZero.self_play_step!(env, Handler(n)) # Use CUDA.@timed to sync up calls
end

for w in workers()
  @spawnat w begin
    println("Running on $(Threads.nthreads()) threads.")
    println("Playing one game to compile everything.")
    bench_self_play(1)
    println("Starting benchmark.")
    report, t, mem, gct = bench_self_play(NUM_GAMES)

    println("Total time: $t")
    println("Spent in GC: $gct")
  end
end
