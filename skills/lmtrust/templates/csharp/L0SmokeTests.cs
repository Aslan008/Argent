// templates/csharp/L0SmokeTests.cs — Layer 0: Smoke & Assembly Load Test
using Xunit;
using FluentAssertions;

namespace LMTrust.Templates.CSharp;

public class L0SmokeTests
{
    [Fact]
    public void Test_L0_Smoke_InstanceCreation()
    {
        var service = new TargetService();
        service.Should().NotBeNull();
        var status = service.GetHealth();
        status.Should().BeTrue();
    }
}
