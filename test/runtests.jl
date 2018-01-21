using SparseMatrixLILs
using Base.Test

    
function test1()
    x0 = sprand(10,15,0.3)
    x = SparseMatrixLIL(x0)
    k = sprand(4,6,0.6)
    x[1:4,1:6] = k
    x0[1:4,1:6] = k
    @test all(x0.==x)
    @test findnz(x0) == findnz(x)
    x[6:9,6:11] = x[6:9,6:11] + k
    x0[6:9,6:11] = x0[6:9,6:11] + k
    @test all(x0.==x)
    @test findnz(x0) == findnz(x)
    x[6:9,6:11] = x[1:4,1:6] + k
    x0[6:9,6:11] = x0[1:4,1:6] + k
    @test all(x0.==x)
    @test findnz(x0) == findnz(x)
    @test vecnorm((x.+sin.(x)) .- (x0.+sin.(x0))) == 0
end

test1()
