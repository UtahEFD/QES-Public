/** QUICFire spread rate using Balbi (2020) model
*This class stores fuel properties of Anderson 13 / Scott and Burgan 40 fuels
**/
class FuelProperties {
    
    public:
        float oneHour;			///< One hour fine fuel load [t/ac]
        float tenHour;			///< Ten hour cured fuel load [t/ac]
        float hundredHour;		///< Hundred hour cured fuel load [t/ac]
        float liveHerb;			///< Live herbacious fuel load [t/ac]
        float liveWoody;		///< Live woody fuel load [t/ac]
        float savOneHour;		///< Surface are to volume ratio one hour fuels [1/ft]
        float savHerb;			///< Surface are to volume ratio herbacious fuels [1/ft]
        float savWoody;			///< Surface are to volume ratio woody fuels [1/ft]
        float fuelDepth;		///< Fuel bed depth [ft]
        float heatContent;		///< Heat Content of fuels [BTU/lb]
        float fuelDensity;		///< Ovendry fuel particle density [lb/ft^3]
		float fuelmce;			///< Dead fuel extinction moisture [%]

        
        FuelProperties(float oneHour, float tenHour, float hundredHour, float liveHerb, float liveWoody, 
                       float savOneHour, float savHerb, float savWoody, float fuelDepth, float heatContent, 
                       float fuelDensity, float fuelmce) : 
                       oneHour(oneHour), tenHour(tenHour), hundredHour(hundredHour), liveHerb(liveHerb), liveWoody(liveWoody), 
                       savOneHour(savOneHour), savHerb(savHerb), savWoody(savWoody), fuelDepth(fuelDepth), heatContent(heatContent), 
                       fuelDensity(fuelDensity), fuelmce(fuelmce) {}
    private:
};
/** 
* Anderson 13 Fuel Classes
**/
// Short grass (1 ft)
class ShortGrass : public FuelProperties {
    public:
        ShortGrass() : FuelProperties(0.7405, 0, 0, 0, 0, 3500, 0, 0, 1, 7496.2, 32.0, 12) {}
};
// Timber (grass and understory)
class TimberGrass : public FuelProperties {
    public:
        TimberGrass() : FuelProperties(4.0015, 0, 0, 0, 0, 2784, 0, 0, 1, 7496.2, 32.0, 15) {}
};
// Tall grass (2.5 ft)
class TallGrass : public FuelProperties {
    public:
        TallGrass() : FuelProperties(3.0111, 0, 0, 0, 0, 1500, 0, 0, 2.5, 7496.2, 32.0, 25) {}
};
// Chaparral (6 ft)
class Chaparral : public FuelProperties {
    public:
        Chaparral() : FuelProperties(11.0098, 0, 0, 0, 0, 1739, 0, 0, 6, 7496.2, 32.0, 20) {}
};
// Brush (2 ft)
class Brush : public FuelProperties {
    public:
        Brush() : FuelProperties(3.5019, 0, 0, 0, 0, 1683, 0, 0, 2, 7496.2, 32.0, 20) {}
};
// Dormant brush, hardwood slash
class DormantBrush : public FuelProperties {
    public:
        DormantBrush() : FuelProperties(6.0000, 0, 0, 0, 0, 1564, 0, 0, 2.5, 7496.2, 32.0, 25) {}
};
// Southern rough
class SouthernRough : public FuelProperties {
    public:
        SouthernRough() : FuelProperties(4.8714, 0, 0, 0, 0, 1562, 0, 0, 2.5, 7496.2, 32.0, 40) {}
};
// Closed timber litter
class TimberClosedLitter : public FuelProperties {
    public:
        TimberClosedLitter() : FuelProperties(5.0008, 0, 0, 0, 0, 1884, 0, 0, 0.2, 7496.2, 32.0, 30) {}
};
// Hardwood litter
class HarwoodLitter : public FuelProperties {
    public:
        HarwoodLitter() : FuelProperties(4.3718, 0, 0, 0, 0, 2484, 0, 0, 0.2, 7496.2, 32.0, 25) {}
};
// Timber (litter + understory)
class TimberLitter : public FuelProperties {
    public:
        TimberLitter() : FuelProperties(12.0179, 0, 0, 0, 0, 1764, 0, 0, 1, 7496.2, 32.0, 25) {}
};
// Light logging slash
class LoggingSlashLight : public FuelProperties {
    public:
        LoggingSlashLight() : FuelProperties(11.5183, 0, 0, 0, 0, 1182, 0, 0, 1, 7496.2, 32.0, 15) {}
};
// Medium logging slash
class LoggingSlashMedium : public FuelProperties {
    public:
        LoggingSlashMedium() : FuelProperties(34.5683, 0, 0, 0, 0, 1145, 0, 0, 2.3, 7496.2, 32.0, 20)  {}
};
// Heavy logging slash
class LoggingSlashHeavy : public FuelProperties {
    public:
        LoggingSlashHeavy() : FuelProperties(51.1000, 0, 0, 0, 0, 1159, 0, 0, 4.1, 7496.2, 32.0, 25) {}
};

/**
* Non-burnable areas
**/

// Urban/Developed
class Urban : public FuelProperties {
    public:
        Urban() : FuelProperties(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) {}
};
// Snow/Ice
class Snow : public FuelProperties {
    public:
        Snow() : FuelProperties(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) {}
};
// Agricultural
class Agricultural : public FuelProperties {
    public:
        Agricultural() : FuelProperties(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) {}
};
// Open Water
class Water : public FuelProperties {
    public:
        Water() : FuelProperties(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) {}
};
// Bare Ground
class Bare : public FuelProperties {
    public:
        Bare() : FuelProperties(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) {}
};

/**
* Scott and Burgan 2005 Fuel Classes
**/

// GR1: Short Sparse, Dry Climate Grass
class GR1 : public FuelProperties {

	public:
		GR1() : FuelProperties(0.10, 0.00, 0.00, 0.30, 0.00, 2200, 2000, 9999, 0.4, 8000, 32.00, 15) {}
};
 
// GR2: Low Load, Dry Climate Grass
class GR2 : public FuelProperties {
	public:
		GR2() : FuelProperties(0.10, 0.00, 0.00, 1.00, 0.00, 2000, 1800, 9999, 1.0, 8000, 32.00, 15) {}
};

// GR3: Low Load, Very Coarse Humid Climate Grass
class GR3 : public FuelProperties {
	public:
		GR3() : FuelProperties(0.10, 0.40, 0.00, 1.50, 0, 1500, 1300, 9999, 2.0, 8000, 32.00, 30) {}
};
 
// GR4: Moderate Load, Dry Climate Grass
class GR4 : public FuelProperties {
	public:
		GR4() : FuelProperties(0.25, 0.00, 0.00, 1.90, 0, 2000, 1800, 9999, 2.0, 8000, 32.00, 15) {}
};
 
// GR5: Low Load, Humid Climate Grass
class GR5 : public FuelProperties {
	public:
		GR5() : FuelProperties(0.40, 0.00, 0.00, 2.50, 0, 1800, 1600, 9999, 1.5, 8000, 32.00, 40) {}
}; 
 
// GR6: Moderate Load, Humid Climate Grass
class GR6 : public FuelProperties {
	public:
		GR6() : FuelProperties(0.10, 0.00, 0.00, 3.40, 0, 2200, 2000, 9999, 1.5, 9000, 32.00, 40) {}
};

// GR7: High Load, Dry Climate Grass
class GR7 : public FuelProperties {
	public:
		GR7() : FuelProperties(1.00, 0.00, 0.00, 5.40, 0, 2000, 1800, 9999, 3.0, 8000, 32.00, 15) {}
};

// GR8: High Load, Very Coarse, Humid Climate Grass
class GR8 : public FuelProperties {
	public:
		GR8() : FuelProperties(0.50, 1.00, 0.00, 7.30, 0, 1500, 1300, 9999, 4.0, 8000, 32.00, 30) {}
};

// GR9: Very High Load, Humid Climate Grass
class GR9 : public FuelProperties {
	public:
		GR9() : FuelProperties(1.00, 1.00, 0.00, 9.00, 0, 1800, 1600, 9999, 5.0, 8000, 32.00, 40) {}
};

// GS1: Low Load, Dry Climate Grass-Shrub
class GS1 : public FuelProperties {
	public:
		GS1() : FuelProperties(0.20, 0.00, 0.00, 0.50, 0.65, 2000, 1800, 1800, 0.9, 8000, 32.00, 15) {}
};

// GS2: Moderate Load, Dry Climate Grass-Shrub
class GS2 : public FuelProperties {
	public:
		GS2() : FuelProperties(0.50, 0.50, 0.00, 0.60, 1.00, 2000, 1800, 1800, 1.5, 8000, 32.0, 15) {}
};

// GS3: Moderate Load, Humid Climate Grass-Shrub
class GS3 : public FuelProperties {
	public:
		GS3() : FuelProperties(0.30, 0.25, 0.00, 1.45, 1.25, 1800, 1600, 1600, 1.8, 8000, 32.00, 40) {}
};

// GS4: High Load, Humid Climate Grass-Shrub
class GS4 : public FuelProperties {
	public:
		GS4() : FuelProperties(1.90, 0.30, 0.10, 3.40, 7.10, 1800, 1600, 1600, 2.1, 8000, 32.00, 40) {}
};

// SH1: Low Load, Dry Climate Shrub
class SH1 : public FuelProperties {
	public:
		SH1() : FuelProperties(0.25, 0.25, 0.00, 0.15, 1.30, 2000, 1800, 1600, 1.0, 8000, 32.00, 15) {}
};

// SH2: Moderate Load, Dry Climate Shrub
class SH2 : public FuelProperties {
	public:
		SH2() : FuelProperties(1.35, 2.40, 0.75, 0.00, 3.85, 2000, 9999, 1600, 1.0, 8000, 32.00, 15) {}
};

// SH3: Moderate Load, Dry Climate Shrub
class SH3 : public FuelProperties {
	public:
		SH3() : FuelProperties(0.45, 3.00, 0.00, 0.00, 6.20, 1600, 9999, 1400, 2.4, 8000, 32.00, 40) {}
};

// SH4: Low Load, Humid Climate Timber-Shrub
class SH4 : public FuelProperties {
	public:
		SH4() : FuelProperties(0.85, 1.15, 0.20, 0.00, 2.55, 2000, 1800, 1600, 3.0, 8000, 32.00, 30) {}
};

// SH5: High Load, Dry Climate Shrub
class SH5 : public FuelProperties {
	public:
		SH5() : FuelProperties(3.60, 2.10, 0.00, 0.00, 2.90, 750, 9999, 1600, 6.0, 8000, 32.00, 15) {}
};

// SH6: Low Load, Humid Climate Shrub
class SH6 : public FuelProperties {
	public:
		SH6() : FuelProperties(2.90, 1.45, 0.00, 0.00, 1.40, 750, 9999, 1600, 2.0, 8000, 32.00, 30) {}
};

// SH7: Very High Load, Dry Climate Shrub
class SH7 : public FuelProperties {
	public:
		SH7() : FuelProperties(3.50, 5.30, 2.20, 0.00, 3.40, 750, 9999, 1600, 6.0, 8000, 32.00, 15) {}
};

// SH8: High Load, Humid Climate Shrub
class SH8 : public FuelProperties {
	public:
		SH8() : FuelProperties(2.05, 3.40, 0.85, 0.00, 4.35, 750, 9999, 1600, 3.0, 8000, 32.00, 40) {}
};

// SH9: Very High Load, Humid Climate Shrub
class SH9 : public FuelProperties {
	public:
		SH9() : FuelProperties(4.50, 2.45, 0.00, 1.55, 7.00, 750, 1800, 1500, 4.4, 8000, 32.00, 40) {}
};

// TU1: Low Load, Dry Climate Timber-Grass-Shrub
class TU1 : public FuelProperties {
	public:
		TU1() : FuelProperties(0.20, 0.90, 1.50, 0.20, 0.90, 2000, 1800, 1600, 0.6, 8000, 32.00, 20) {}
};

// TU2: Moderate Load, Humid Climate Timber-Shrub
class TU2 : public FuelProperties {
	public:
		TU2() : FuelProperties(0.95, 1.80, 1.25, 0.00, 0.20, 2000, 9999, 1600, 1.0, 8000, 32.00, 30) {}
};

// TU3: Moderate Load, Humid Climate Timber-Grass-Shrub
class TU3 : public FuelProperties {
	public:
		TU3() : FuelProperties(1.10, 0.15, 0.25, 0.65, 1.10, 1800, 1600, 1400, 1.3, 8000, 32.00, 30) {}
};

// TU4: Dwarf Conifer with Understory
class TU4 : public FuelProperties {
	public:
		TU4() : FuelProperties(4.50, 0.00, 0.00, 0.00, 2.00, 2300, 9999, 2000, 0.5, 8000, 32.00, 12) {}
};

// TU5: Very High Load, Dry Climate Timber-Shrub
class TU5 : public FuelProperties {
	public:
		TU5() : FuelProperties(4.00, 4.00, 3.00, 0.00, 3.00, 1500, 9999, 750, 1.0, 8000, 32.00, 25) {}
};

// TL1: Low Load Compact Conifer Litter
class TL1 : public FuelProperties {
	public:
		TL1() : FuelProperties(1.00, 2.20, 3.60, 0.00, 0.00, 2000, 9999, 9999, 0.2, 8000, 32.00, 30) {}
};

// TL2: Low Load Broadleaf Litter
class TL2 : public FuelProperties {
	public:
		TL2() : FuelProperties(1.40, 2.30, 2.20, 0.00, 0.00, 2000, 9999, 9999, 0.2, 8000, 32.00, 25) {}
};

// TL3: Moderate Load Conifer Litter
class TL3 : public FuelProperties {
	public:
		TL3() : FuelProperties(0.50, 2.20, 2.80, 0.00, 0.00, 2000, 9999, 9999, 0.3, 8000, 32.00, 20) {}
};

// TL4: Small Downed Logs
class TL4 : public FuelProperties {
	public:
		TL4() : FuelProperties(0.50, 1.50, 4.20, 0.00, 0.00, 2000, 9999, 9999, 0.4, 8000, 32.00, 25) {}
};

// TL5: High Load Conifer Litter
class TL5 : public FuelProperties {
	public:
		TL5() : FuelProperties(1.15, 2.50, 4.40, 0.00, 0.00, 2000, 9999, 9999, 0.6, 8000, 32.00, 25) {}
};

// TL6: Moderate Load Broadleaf Litter
class TL6 : public FuelProperties {
	public:
		TL6() : FuelProperties(2.40, 1.20, 1.20, 0.00, 0.00, 2000, 9999, 9999, 0.3, 8000, 32.00, 25) {}
};

// TL7: Large Downed Logs
class TL7 : public FuelProperties {
	public:
		TL7() : FuelProperties(0.30, 1.40, 8.10, 0.00, 0.00, 2000, 9999, 9999, 0.4, 8000, 32.00, 25) {}
};

// TL8: Long-Needle Litter
class TL8 : public FuelProperties {
	public:
		TL8() : FuelProperties(5.80, 1.40, 1.10, 0.00, 0.00, 1800, 9999, 9999, 0.3, 8000, 32.00, 35) {}
};

//TL9: Very High Load Broadleaf Litter
class TL9 : public FuelProperties {
	public:
		TL9() : FuelProperties(6.65, 3.30, 4.15, 0.00, 0.00, 1800, 9999, 1600, 0.6, 8000, 32.00, 35) {}
};

// SB1: Low Load Activity Fuel
class SB1 : public FuelProperties {
	public:
		SB1() : FuelProperties(1.50, 3.00, 11.00, 0.00, 0.00, 2000, 9999, 9999, 1.0, 8000, 32.00, 25) {}
};

// SB2: Moderate Load Activity Fuel or Low Load Blowdown
class SB2 : public FuelProperties {
	public:
		SB2() : FuelProperties(4.50, 4.25, 4.00, 0.00, 0.00, 2000, 9999, 9999, 1.0, 8000, 32.00, 25) {}
};
 
// SB3: High Load Activity Fuel or Moderate Load Blowdown
class SB3 : public FuelProperties {
	public:
		SB3() : FuelProperties(5.50, 2.75, 3.00, 0.00, 0.00, 2000, 9999, 9999, 1.2, 8000, 32.00, 25) {}
};

// SB4: High Load Blowdown
class SB4 : public FuelProperties {
	public:
		SB4() : FuelProperties(5.25, 3.50, 5.25, 0.00, 0.00, 2000, 9999, 9999, 2.7, 8000, 32.00, 25) {}
};