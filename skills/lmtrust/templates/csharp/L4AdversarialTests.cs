// templates/csharp/L4AdversarialTests.cs — Layer 4: Adversarial error handling
using Xunit;
using FluentAssertions;
using System;

namespace LMTrust.Templates.CSharp;

public class L4AdversarialTests
{
    [Fact]
    public void Test_L4_Adversarial_ZeroDivisionThrowsControlled()
    {
        var engine = new CombatEngine();
        Action act = () => engine.ApplyDamageMultiplier(100, 0);

        // Must throw ArgumentException rather than unhandled DivideByZeroException
        act.Should().Throw<ArgumentException>()
           .WithMessage("*divisor cannot be zero*");
    }
}
