// QUICFire spread rate using Balbi (2020) model
// This class stores fuel properties of Scott and Burgan 40 fuels

class FuelProperties40 {
    
    public:
        float oneHour;
        float tenHour;
        float hundredHour;
        float liveHerb;
        float liveWoody;
        float savOneHour;
        float savHerb;
        float savWoody;
        float fuelDepth;
        float heatContent;
        float fuelDensity;

        
        FuelProperties40(float oneHour, float tenHour, float hundredHour, float liveHerb, float liveWoody, 
                       float savOneHour, float savHerb, float savWoody, float fuelDepth, float heatContent, 
                       float fuelDensity) : 
                       oneHour(oneHour), tenHour(tenHour), hundredHour(hundredHour), liveHerb(liveHerb), liveWoody(liveWoody), 
                       savOneHour(savOneHour), savHerb(savHerb), savWoody(savWoody), fuelDepth(fuelDepth), heatContent(heatContent), 
                       fuelDensity(fuelDensity) {}
    private:
};

// GR1: Short Sparse, Dry Climate Grass
class GR1 : public FuelProperties40 {
	public:
		GR1() : FuelProperties40(0.10, 0.00, 0.00, 0.30, 0.00, 2200, 2000, 9999, 0.4, 8000, 32.00) {}
};
 
// GR2: Low Load, Dry Climate Grass
class GR2 : public FuelProperties40 {
	public:
		GR2() : FuelProperties40(0.10, 0.00, 0.00, 1.00, 0.00, 2000, 1800, 9999, 1.0, 8000, 32.00) {}
};

// GR3: Low Load, Very Coarse Humid Climate Grass
class GR3 : public FuelProperties40 {
	public:
		GR3() : FuelProperties40(0.10, 0.40, 0.00, 1.50, 0, 1500, 1300, 9999, 2.0, 8000, 32.00) {}
};
 
// GR4: Moderate Load, Dry Climate Grass
class GR4 : public FuelProperties40 {
	public:
		GR4() : FuelProperties40(0.25, 0.00, 0.00, 1.90, 0, 2000, 1800, 9999, 2.0, 8000, 32.00) {}
};
 
// GR5: Low Load, Humid Climate Grass
class GR5 : public FuelProperties40 {
	public:
		GR5() : FuelProperties40(0.40, 0.00, 0.00, 2.50, 0, 1800, 1600, 9999, 1.5, 8000, 32.00) {}
}; 
 
// GR6: Moderate Load, Humid Climate Grass
class GR6 : public FuelProperties40 {
	public:
		GR6() : FuelProperties40(0.10, 0.00, 0.00, 3.40, 0, 2200, 2000, 9999, 1.5, 9000, 32.00) {}
};

// GR7: High Load, Dry Climate Grass
class GR7 : public FuelProperties40 {
	public:
		GR7() : FuelProperties40(1.00, 0.00, 0.00, 5.40, 0, 2000, 1800, 9999, 3.0, 8000, 32.00) {}
};

// GR8: High Load, Very Coarse, Humid Climate Grass
class GR8 : public FuelProperties40 {
	public:
		GR8() : FuelProperties40(0.50, 1.00, 0.00, 7.30, 0, 1500, 1300, 9999, 4.0, 8000, 32.00) {}
};

// GR9: Very High Load, Humid Climate Grass
class GR9 : public FuelProperties40 {
	public:
		GR9() : FuelProperties40(1.00, 1.00, 0.00, 9.00, 0, 1800, 1600, 9999, 5.0, 8000, 32.00) {}
};

// GS1: Low Load, Dry Climate Grass-Shrub
class GS1 : public FuelProperties40 {
	public:
		GS1() : FuelProperties40(0.20, 0.00, 0.00, 0.50, 0.65, 2000, 1800, 1800, 0.9, 8000, 32.00) {}
};

// GS2: Moderate Load, Dry Climate Grass-Shrub
class GS2 : public FuelProperties40 {
	public:
		GS2() : FuelProperties40(0.50, 0.50, 0.00, 0.60, 1.00, 2000, 1800, 1800, 1.5, 8000, 32.0) {}
};

// GS3: Moderate Load, Humid Climate Grass-Shrub
class GS3 : public FuelProperties40 {
	public:
		GS3() : FuelProperties40(0.30, 0.25, 0.00, 1.45, 1.25, 1800, 1600, 1600, 1.8, 8000, 32.00) {}
};

// GS4: High Load, Humid Climate Grass-Shrub
class GS4 : public FuelProperties40 {
	public:
		GS4() : FuelProperties40(1.90, 0.30, 0.10, 3.40, 7.10, 1800, 1600, 1600, 2.1, 8000, 32.00) {}
};

// SH1: Low Load, Dry Climate Shrub
class SH1 : public FuelProperties40 {
	public:
		SH1() : FuelProperties40(0.25, 0.25, 0.00, 0.15, 1.30, 2000, 1800, 1600, 1.0, 8000, 32.00) {}
};

// SH2: Moderate Load, Dry Climate Shrub
class SH2 : public FuelProperties40 {
	public:
		SH2() : FuelProperties40(1.35, 2.40, 0.75, 0.00, 3.85, 2000, 9999, 1600, 1.0, 8000, 32.00) {}
};

// SH3: Moderate Load, Dry Climate Shrub
class SH3 : public FuelProperties40 {
	public:
		SH3() : FuelProperties40(0.45, 3.00, 0.00, 0.00, 6.20, 1600, 9999, 1400, 2.4, 8000, 32.00) {}
};

// SH4: Low Load, Humid Climate Timber-Shrub
class SH4 : public FuelProperties40 {
	public:
		SH4() : FuelProperties40(0.85, 1.15, 0.20, 0.00, 2.55, 2000, 1800, 1600, 3.0, 8000, 32.00) {}
};

// SH5: High Load, Dry Climate Shrub
class SH5 : public FuelProperties40 {
	public:
		SH5() : FuelProperties40(3.60, 2.10, 0.00, 0.00, 2.90, 750, 9999, 1600, 6.0, 8000, 32.00) {}
};

// SH6: Low Load, Humid Climate Shrub
class SH6 : public FuelProperties40 {
	public:
		SH6() : FuelProperties40(2.90, 1.45, 0.00, 0.00, 1.40, 750, 9999, 1600, 2.0, 8000, 32.00) {}
};

// SH7: Very High Load, Dry Climate Shrub
class SH7 : public FuelProperties40 {
	public:
		SH7() : FuelProperties40(3.50, 5.30, 2.20, 0.00, 3.40, 750, 9999, 1600, 6.0, 8000, 32.00) {}
};

// SH8: High Load, Humid Climate Shrub
class SH8 : public FuelProperties40 {
	public:
		SH8() : FuelProperties40(2.05, 3.40, 0.85, 0.00, 4.35, 750, 9999, 1600, 3.0, 8000, 32.00) {}
};

// SH9: Very High Load, Humid Climate Shrub
class SH9 : public FuelProperties40 {
	public:
		SH9() : FuelProperties40(4.50, 2.45, 0.00, 1.55, 7.00, 750, 1800, 1500, 4.4, 8000, 32.00) {}
};

// TU1: Low Load, Dry Climate Timber-Grass-Shrub
class TU1 : public FuelProperties40 {
	public:
		TU1() : FuelProperties40(0.20, 0.90, 1.50, 0.20, 0.90, 2000, 1800, 1600, 0.6, 8000, 32.00) {}
};

// TU2: Moderate Load, Humid Climate Timber-Shrub
class TU2 : public FuelProperties40 {
	public:
		TU2() : FuelProperties40(0.95, 1.80, 1.25, 0.00, 0.20, 2000, 9999, 1600, 1.0, 8000, 32.00) {}
};

// TU3: Moderate Load, Humid Climate Timber-Grass-Shrub
class TU3 : public FuelProperties40 {
	public:
		TU3() : FuelProperties40(1.10, 0.15, 0.25, 0.65, 1.10, 1800, 1600, 1400, 1.3, 8000, 32.00) {}
};

// TU4: Dwarf Conifer with Understory
class TU4 : public FuelProperties40 {
	public:
		TU4() : FuelProperties40(4.50, 0.00, 0.00, 0.00, 2.00, 2300, 9999, 2000, 0.5, 8000, 32.00) {}
};

// TU5: Very High Load, Dry Climate Timber-Shrub
class TU5 : public FuelProperties40 {
	public:
		TU5() : FuelProperties40(4.00, 4.00, 3.00, 0.00, 3.00, 1500, 9999, 750, 1.0, 8000, 32.00) {}
};

// TL1: Low Load Compact Conifer Litter
class TL1 : public FuelProperties40 {
	public:
		TL1() : FuelProperties40(1.00, 2.20, 3.60, 0.00, 0.00, 2000, 9999, 9999, 0.2, 8000, 32.00) {}
};

// TL2: Low Load Broadleaf Litter
class TL2 : public FuelProperties40 {
	public:
		TL2() : FuelProperties40(1.40, 2.30, 2.20, 0.00, 0.00, 2000, 9999, 9999, 0.2, 8000, 32.00) {}
};

// TL3: Moderate Load Conifer Litter
class TL3 : public FuelProperties40 {
	public:
		TL3() : FuelProperties40(0.50, 2.20, 2.80, 0.00, 0.00, 2000, 9999, 9999, 0.3, 8000, 32.00) {}
};

// TL4: Small Downed Logs
class TL4 : public FuelProperties40 {
	public:
		TL4() : FuelProperties40(0.50, 1.50, 4.20, 0.00, 0.00, 2000, 9999, 9999, 0.4, 8000, 32.00) {}
};

// TL5: High Load Conifer Litter
class TL5 : public FuelProperties40 {
	public:
		TL5() : FuelProperties40(1.15, 2.50, 4.40, 0.00, 0.00, 2000, 9999, 9999, 0.6, 8000, 32.00) {}
};

// TL6: Moderate Load Broadleaf Litter
class TL6 : public FuelProperties40 {
	public:
		TL6() : FuelProperties40(2.40, 1.20, 1.20, 0.00, 0.00, 2000, 9999, 9999, 0.3, 8000, 32.00) {}
};

// TL7: Large Downed Logs
class TL7 : public FuelProperties40 {
	public:
		TL7() : FuelProperties40(0.30, 1.40, 8.10, 0.00, 0.00, 2000, 9999, 9999, 0.4, 8000, 32.00) {}
};

// TL8: Long-Needle Litter
class TL8 : public FuelProperties40 {
	public:
		TL8() : FuelProperties40(5.80, 1.40, 1.10, 0.00, 0.00, 1800, 9999, 9999, 0.3, 8000, 32.00) {}
};

//TL9: Very High Load Broadleaf Litter
class TL9 : public FuelProperties40 {
	public:
		TL9() : FuelProperties40(6.65, 3.30, 4.15, 0.00, 0.00, 1800, 9999, 1600, 0.6, 8000, 32.00) {}
};

// SB1: Low Load Activity Fuel
class SB1 : public FuelProperties40 {
	public:
		SB1() : FuelProperties40(1.50, 3.00, 11.00, 0.00, 0.00, 2000, 9999, 9999, 1.0, 8000, 32.00) {}
};

// SB2: Moderate Load Activity Fuel or Low Load Blowdown
class SB2 : public FuelProperties40 {
	public:
		SB2() : FuelProperties40(4.50, 4.25, 4.00, 0.00, 0.00, 2000, 9999, 9999, 1.0, 8000, 32.00) {}
};
 
// SB3: High Load Activity Fuel or Moderate Load Blowdown
class SB3 : public FuelProperties40 {
	public:
		SB3() : FuelProperties40(5.50, 2.75, 3.00, 0.00, 0.00, 2000, 9999, 9999, 1.2, 8000, 32.00) {}
};

// SB4: High Load Blowdown
class SB4 : public FuelProperties40 {
	public:
		SB4() : FuelProperties40(5.25, 3.50, 5.25, 0.00, 0.00, 2000, 9999, 9999, 2.7, 8000, 32.00) {}
};