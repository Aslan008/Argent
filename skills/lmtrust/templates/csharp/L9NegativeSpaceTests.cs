// templates/csharp/L9NegativeSpaceTests.cs — Layer 9: Forbidden Side Effects
using Xunit;
using FluentAssertions;
using System.Collections.Generic;

namespace LMTrust.Templates.CSharp;

public class L9NegativeSpaceTests
{
    [Fact]
    public void Test_L9_NegativeSpace_ReadOnlyValidationDoesNotMutate()
    {
        var original = new List<int> { 1, 2, 3 };
        var validator = new ListValidator();

        validator.Validate(original);

        // Validation must not mutate element order or collection items
        original.Should().ContainInOrder(1, 2, 3);
        original.Should().HaveCount(3);
    }
}
