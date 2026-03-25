def main():
    """Test the dynamic scout system"""
    print("=" * 70)
    print("DYNAMIC SCOUT SYSTEM - TEST")
    print("=" * 70)
    
    # Create scout system
    scout = DynamicScoutSystem()
    
    # Print summary
    scout.print_summary()
    
    # Get top 12 assets
    print("\n[TOP 12 SELECTED FOR TRADING]")
    top_12 = scout.get_top_assets(n=12)
    for i, asset in enumerate(top_12, 1):
        print(f"  {i}. {asset.symbol} (Tier: {asset.tier}, Score: {asset.total_score:.1f})")
    
    # Get top 24 for wider selection
    print("\n[TOP 24 BACKUP]")
    top_24 = scout.get_top_assets(n=24)
    for i, asset in enumerate(top_24, 1):
        print(f"  {i}. {asset.symbol} (Tier: {asset.tier})")


if __name__ == '__main__':
    main()
