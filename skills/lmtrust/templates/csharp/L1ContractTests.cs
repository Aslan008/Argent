// templates/csharp/L1ContractTests.cs — Layer 1: Contract & Known Coordinates
using Xunit;
using FluentAssertions;

namespace LMTrust.Templates.CSharp;

public class L1ContractTests
{
    [Theory]
    [InlineData(100, 20, 120)]
    [InlineData(0, 50, 50)]
    public void Test_L1_Contract_KnownCoordinates(int baseVal, int addition, int expected)
    {
        var calculator = new FeeCalculator();
        var actual = calculator.ComputeTotal(baseVal, addition);
        actual.Should().Be(expected);
    }
}
