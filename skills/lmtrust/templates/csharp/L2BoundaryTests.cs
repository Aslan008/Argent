// templates/csharp/L2BoundaryTests.cs — Layer 2: 8 Boundary Conditions
using Xunit;
using FluentAssertions;

namespace LMTrust.Templates.CSharp;

public class L2BoundaryTests
{
    [Fact]
    public void Test_L2_Boundary_NumericLimits()
    {
        var processor = new NumericProcessor();

        // 1. Zero
        processor.Process(0).Should().Be(0);

        // 2. Int32 Min
        processor.Invoking(p => p.Process(int.MinValue)).Should().NotThrow<NullReferenceException>();

        // 3. Int32 Max
        processor.Invoking(p => p.Process(int.MaxValue)).Should().NotThrow<NullReferenceException>();
    }
}
