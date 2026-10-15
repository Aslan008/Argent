// templates/csharp/L3PropertyTests.cs — Layer 3: Property testing with FsCheck
using FsCheck;
using FsCheck.Xunit;
using FluentAssertions;

namespace LMTrust.Templates.CSharp;

public class L3PropertyTests
{
    [Property]
    public Property Test_L3_Property_Monotonicity(NonNegativeInt a, NonNegativeInt b)
    {
        var x = a.Get;
        var y = b.Get;
        var calc = new OrderScorer();

        return (x <= y).Implies(() => calc.Score(x) <= calc.Score(y));
    }
}
