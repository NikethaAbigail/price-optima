import React, { useState, useCallback } from 'react';
import axios from 'axios';
// Importing new, more attractive icons from lucide-react
import { DollarSign, Cpu, Clock, MapPin, Users, Gauge, TrendingUp, BarChart3, LineChart, Activity, Minimize, Maximize } from 'lucide-react'; 
import { BarChart, Bar, LineChart as RechartLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// Define the base URL for your FastAPI server
const API_BASE_URL = 'http://localhost:8000';

// Mock data for the dashboard (will be updated when the real recommendation is fetched)
const initialDashboardData = {
    priceDistribution: [
        { name: '₹0-200', count: 150 },
        { name: '₹201-400', count: 450 },
        { name: '₹401-600', count: 250 },
        { name: '₹601+', count: 80 },
    ],
    performanceMetrics: [
        { time: 'T-3m', Price: 320, Completion: 0.78, GM: 0.35 },
        { time: 'T-2m', Price: 350, Completion: 0.81, GM: 0.37 },
        { time: 'T-1m', Price: 364, Completion: 0.82, GM: 0.38 },
        { time: 'Current', Price: 364.78, Completion: 0.82, GM: 0.38 },
    ],
};

// Helper component for styled input fields
const InputField = React.memo(({ label, name, type = "text", value, onChange, options, min, max, step, icon: Icon }) => {
    const isSelect = Array.isArray(options);

    return (
        <div className="flex flex-col space-y-1">
            <label htmlFor={name} className="text-sm font-semibold text-gray-700 flex items-center">
                {Icon && <Icon className="w-4 h-4 mr-2 text-blue-600" />}
                {label}
            </label>
            <div className="relative">
                {isSelect ? (
                    <select
                        id={name}
                        name={name}
                        value={value}
                        onChange={onChange}
                        className="p-3 border-2 border-gray-200 rounded-xl shadow-sm focus:ring-blue-500 focus:border-blue-500 transition duration-200 ease-in-out bg-white w-full appearance-none"
                    >
                        {options.map(option => (
                            <option key={option} value={option}>
                                {option}
                            </option>
                        ))}
                    </select>
                ) : (
                    <input
                        id={name}
                        name={name}
                        type={type === 'integer' ? 'number' : type}
                        value={value}
                        onChange={onChange}
                        min={min}
                        max={max}
                        step={step}
                        className="p-3 border-2 border-gray-200 rounded-xl shadow-sm focus:ring-blue-500 focus:border-blue-500 transition duration-200 ease-in-out w-full"
                        placeholder={`Enter ${label}`}
                    />
                )}
            </div>
        </div>
    );
});

// New component for comparing the recommended price against a reference price
const PriceComparisonBadge = React.memo(({ label, referencePrice, recommendedPrice, isCompetitor }) => {
    const recommended = parseFloat(recommendedPrice);
    const reference = parseFloat(referencePrice);
    
    // Check for valid numbers
    if (isNaN(recommended) || isNaN(reference)) {
        return <div className="p-4 rounded-xl shadow-inner border border-gray-300 bg-gray-50"><p className="text-sm font-semibold text-gray-700">{label}</p><p className="text-xs mt-1 text-gray-500">N/A</p></div>;
    }

    const diff = recommended - reference;
    const diffPct = (diff / reference) * 100;

    let colorClass, trendText, Icon;

    if (isCompetitor) {
        // Strategy: Being LOWER than competitor is generally good for volume/completion (Green).
        if (diff < 0) { 
            colorClass = "text-green-600 bg-green-100 border-green-300"; 
            trendText = "Lower";
            Icon = Minimize;
        } else if (diff > 0) {
            colorClass = "text-red-600 bg-red-100 border-red-300";
            trendText = "Higher";
            Icon = Maximize;
        } else {
            colorClass = "text-gray-600 bg-gray-100 border-gray-300";
            trendText = "Matches";
            Icon = Gauge;
        }
    } else {
        // Strategy: Being HIGHER than historical cost is generally good for GM (Green).
        if (diff > 0) {
            colorClass = "text-green-600 bg-green-100 border-green-300";
            trendText = "Higher";
            Icon = Maximize; 
        } else if (diff < 0) {
            // Use orange for caution if price is below historical (Lower GM)
            colorClass = "text-orange-600 bg-orange-100 border-orange-300";
            trendText = "Lower";
            Icon = Minimize; 
        } else {
            colorClass = "text-gray-600 bg-gray-100 border-gray-300";
            trendText = "Matches";
            Icon = Gauge;
        }
    }

    return (
        <div className={`p-4 rounded-xl shadow-lg border ${colorClass}`}>
            <div className="flex justify-between items-center">
                <p className="text-sm font-semibold text-gray-700">{label}</p>
                <Icon className={`w-5 h-5 ${colorClass.split(' ')[0]}`} />
            </div>
            <p className="text-3xl font-bold mt-1 text-gray-900">₹{referencePrice}</p>
            <div className="mt-2 text-sm flex justify-between items-center">
                <div>
                    <span className={`font-extrabold ${colorClass.split(' ')[0]}`}>{diff > 0 ? '+' : ''}{diff.toFixed(2)}</span>
                    <span className="ml-1 text-gray-500 text-xs">({diffPct.toFixed(1)}%)</span>
                </div>
                <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${colorClass}`.replace('border-', 'border-transparent').replace('shadow-lg', 'shadow-none')}>
                    {trendText}
                </span>
            </div>
        </div>
    );
});


// Dashboard Component using Recharts
const AnalyticsDashboard = React.memo(({ data }) => {
    // Color mapping for charts: Blue (Price), Red (Completion), Orange (GM)
    const chartColors = {
        price: "#3b82f6",     // Blue-500
        completion: "#ef4444", // Red-500
        gm: "#f97316",         // Orange-600
        bar: "#2563eb",        // Primary Blue for Bar Chart
    };

    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-8">
            {/* Chart 1: Price Distribution */}
            <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100">
                <h3 className="text-xl font-bold mb-4 text-gray-800 flex items-center">
                    <BarChart3 className="w-5 h-5 mr-2 text-red-500" /> Optimized Price Distribution (Mock)
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={data.priceDistribution} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                        <XAxis dataKey="name" stroke="#333" />
                        <YAxis stroke="#333" />
                        <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }} />
                        <Bar dataKey="count" fill={chartColors.bar} name="Number of Rides" radius={[10, 10, 0, 0]} />
                    </BarChart>
                </ResponsiveContainer>
                <p className="text-sm text-gray-500 mt-3 text-center">Simulated distribution of recommended prices.</p>
            </div>

            {/* Chart 2: Time Series Performance */}
            <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100">
                <h3 className="text-xl font-bold mb-4 text-gray-800 flex items-center">
                    <LineChart className="w-5 h-5 mr-2 text-orange-500" /> Recent Policy Performance
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                    <RechartLineChart data={data.performanceMetrics} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                        <XAxis dataKey="time" stroke="#333" />
                        <YAxis yAxisId="left" stroke={chartColors.price} label={{ value: 'Price (₹)', angle: -90, position: 'insideLeft', fill: chartColors.price }} />
                        <YAxis yAxisId="right" orientation="right" stroke="#f97316" label={{ value: 'P(Completion) / GM', angle: 90, position: 'insideRight', fill: '#f97316' }} />
                        <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }} />
                        <Legend wrapperStyle={{ paddingTop: '10px' }} />
                        <Line yAxisId="left" type="monotone" dataKey="Price" stroke={chartColors.price} strokeWidth={2} activeDot={{ r: 8 }} /> 
                        <Line yAxisId="right" type="monotone" dataKey="Completion" stroke={chartColors.completion} strokeWidth={2} dot={false} />
                        <Line yAxisId="right" type="monotone" dataKey="GM" stroke={chartColors.gm} strokeWidth={2} dot={false} />
                    </RechartLineChart>
                </ResponsiveContainer>
                <p className="text-sm text-gray-500 mt-3 text-center">How the policy and key metrics have trended over time.</p>
            </div>
        </div>
    );
});


const App = () => {
    // State to manage the visibility of the dashboard
    const [showDashboard, setShowDashboard] = useState(false);
    const [dashboardData, setDashboardData] = useState(initialDashboardData);
    
    // Initial state for all 11 inputs
    const [formData, setFormData] = useState({
        numRiders: 65,
        numDrivers: 25,
        locationCategory: 'Urban',
        loyaltyStatus: 'Silver',
        numPastRides: 30, 
        avgRatings: 4.5,
        timeOfBooking: 'Evening',
        vehicleType: 'Premium',
        expectedDuration: 48,
        historicalCost: 250,
        competitorPrice: 360, 
    });

    const [recommendation, setRecommendation] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleChange = useCallback((e) => {
        const { name, value, type } = e.target;
        const finalValue = type === 'number' ? (parseFloat(value) || 0) : value;

        setFormData(prev => ({
            ...prev,
            [name]: finalValue
        }));
    }, []);

    const handleRecommendation = async () => {
        setShowDashboard(false); // Hide dashboard while processing
        setLoading(true);
        setRecommendation(null);
        setError(null);

        // Basic validation check
        const intFields = ['numRiders', 'numDrivers', 'numPastRides', 'expectedDuration', 'historicalCost', 'competitorPrice'];
        const valid = intFields.every(field => formData[field] >= 0);

        if (!valid) {
            setError("Please ensure all numeric inputs are non-negative numbers.");
            setLoading(false);
            return;
        }

        // Map frontend camelCase state keys to backend snake_case API keys
        const payloadRecord = {
            "Number_of_Riders": formData.numRiders,
            "Number_of_Drivers": formData.numDrivers,
            "Location_Category": formData.locationCategory,
            "Customer_Loyalty_Status": formData.loyaltyStatus,
            "Number_of_Past_Rides": formData.numPastRides,
            "Average_Ratings": formData.avgRatings,
            "Time_of_Booking": formData.timeOfBooking,
            "Vehicle_Type": formData.vehicleType,
            "Expected_Ride_Duration": formData.expectedDuration,
            "Historical_Cost_of_Ride": formData.historicalCost,
            "competitor_price": formData.competitorPrice 
        };
        
        const apiPayload = { record: payloadRecord };

        try {
            const response = await axios.post(`${API_BASE_URL}/recommend`, apiPayload, {
                headers: { 'Content-Type': 'application/json' }
            });
            
            const result = response.data;
            
            const newRecommendation = {
                priceRecommended: result.price_recommended.toFixed(2),
                pCompleteRecommended: result.p_complete_recommended.toFixed(2),
                // GM (Gross Margin) is in percentage, assuming 0.0 to 1.0 scale
                confidence: (result.gm_pct * 100).toFixed(2), 
            };
            setRecommendation(newRecommendation);

            // Update dashboard data with the new recommendation for the 'Current' point
            setDashboardData(prev => {
                const updatedPerformance = [...prev.performanceMetrics];
                // Replace or append the 'Current' entry
                updatedPerformance[updatedPerformance.length - 1] = {
                    time: 'Current', 
                    Price: parseFloat(newRecommendation.priceRecommended), 
                    Completion: parseFloat(newRecommendation.pCompleteRecommended), 
                    GM: parseFloat(newRecommendation.confidence) / 100 // Convert back to 0-1 scale for chart display consistency if needed, but here using raw GM%.
                };
                return {
                    ...prev,
                    performanceMetrics: updatedPerformance
                }
            });

            // Show the dashboard after a successful calculation
            setShowDashboard(true);
            
        } catch (err) {
            console.error("API Error:", err);
            
            let detail;
            if (err.response && err.response.data && err.response.data.detail) {
                detail = JSON.stringify(err.response.data.detail, null, 2);
            } else if (err.message) {
                detail = err.message;
                if (err.message.includes('Network Error')) {
                    detail = "Network error: Cannot connect to the backend server. Please ensure the server is running at " + API_BASE_URL;
                }
            } else {
                detail = "Failed to connect to backend server. Check CORS or server status.";
            }

            setError(`API Request Failed: ${detail}`);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gray-100 flex flex-col items-center p-4 sm:p-8 font-['Inter']"> 
            
            {/* Header - VIBRANT GRADIENT */}
            <header className="w-full max-w-5xl py-8 mb-8 bg-gradient-to-r from-blue-600 to-indigo-700 rounded-2xl shadow-2xl text-white text-center">
                <div className="flex justify-center items-center space-x-4">
                    <Activity className="w-10 h-10 text-orange-300" strokeWidth={2.5}/>
                    <h1 className="text-4xl font-extrabold tracking-tight">
                        PriceOptima
                    </h1>
                </div>
                <p className="text-lg mt-2 text-blue-200 font-medium">
                    AI-Powered Dynamic Pricing Engine
                </p>
            </header>

            {/* Main Content Card - INPUTS */}
            <div className="w-full max-w-5xl bg-white p-8 rounded-3xl shadow-2xl border border-gray-100">
                <h2 className="text-2xl font-bold mb-8 text-gray-800 border-b pb-4 border-blue-100 flex items-center">
                    <Cpu className="w-6 h-6 mr-2 text-blue-600" /> Input Ride Factors
                </h2>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                    {/* --- Inputs --- */}
                    <InputField label="Number of Riders" name="numRiders" type="integer" value={formData.numRiders} onChange={handleChange} min="0" icon={Users} />
                    <InputField label="Number of Drivers" name="numDrivers" type="integer" value={formData.numDrivers} onChange={handleChange} min="0" icon={Users} />
                    <InputField label="Location Category" name="locationCategory" value={formData.locationCategory} onChange={handleChange} options={['Urban', 'Sub-Urban', 'Rural']} icon={MapPin} />
                    <InputField label="Customer Loyalty Status" name="loyaltyStatus" value={formData.loyaltyStatus} onChange={handleChange} options={['Gold', 'Regular', 'Silver']} icon={TrendingUp} />

                    {/* --- Row 2 --- */}
                    <InputField label="Number of Past Rides" name="numPastRides" type="integer" value={formData.numPastRides} onChange={handleChange} min="0" icon={Clock} />
                    <InputField label="Average Ratings" name="avgRatings" type="number" value={formData.avgRatings} onChange={handleChange} min="1.0" max="5.0" step="0.1" icon={Gauge} />
                    <InputField label="Time of Booking" name="timeOfBooking" value={formData.timeOfBooking} onChange={handleChange} options={['Morning', 'Afternoon', 'Evening', 'Night']} icon={Clock} />
                    <InputField label="Vehicle Type" name="vehicleType" value={formData.vehicleType} onChange={handleChange} options={['Economy', 'Premium']} icon={MapPin} />

                    {/* --- Row 3 (Remaining fields) --- */}
                    <InputField label="Expected Ride Duration (mins)" name="expectedDuration" type="integer" value={formData.expectedDuration} onChange={handleChange} min="1" icon={Clock} />
                    <InputField label="Historical Cost of Ride (₹)" name="historicalCost" type="integer" value={formData.historicalCost} onChange={handleChange} min="1" icon={DollarSign} />
                    <InputField label="Competitor Price (₹)" name="competitorPrice" type="integer" value={formData.competitorPrice} onChange={handleChange} min="1" icon={DollarSign} />
                </div>

                {/* Submit Button */}
                <div className="mt-10 col-span-full">
                    <button
                        onClick={handleRecommendation}
                        disabled={loading}
                        className="w-full md:w-auto px-12 py-4 bg-red-500 hover:bg-red-600 text-white font-extrabold text-xl rounded-2xl shadow-lg shadow-red-300/50 transition duration-300 ease-in-out transform hover:scale-[1.02] disabled:bg-red-300 disabled:cursor-not-allowed flex items-center justify-center space-x-3"
                    >
                        {loading && (
                            <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                        )}
                        <span>{loading ? 'Calculating Price...' : 'Get Price Recommendation'}</span>
                    </button>
                </div>
            </div>

            {/* Recommendation Output/Error */}
            <div className="w-full max-w-5xl mt-6">
                {error && (
                    <div className="bg-rose-100 border border-rose-400 text-rose-700 px-4 py-3 rounded-xl relative mb-4 font-mono overflow-x-auto" role="alert">
                        <strong className="font-bold">API Error:</strong>
                        <pre className="block sm:inline ml-2 whitespace-pre-wrap text-sm">{error}</pre>
                    </div>
                )}

                {recommendation && (
                    <div className="bg-blue-50 p-8 rounded-2xl shadow-xl border-4 border-blue-300">
                        {/* Recommendation Header */}
                        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center border-b pb-4 mb-4">
                            <div className="flex items-center space-x-4 mb-4 sm:mb-0">
                                <DollarSign className="w-12 h-12 text-blue-600 animate-pulse" strokeWidth={2.5}/>
                                <div>
                                    <p className="text-lg font-medium text-gray-700">Optimal Recommended Price</p>
                                    <p className="text-5xl font-extrabold text-blue-700 mt-1">
                                        ₹{recommendation.priceRecommended}
                                    </p>
                                </div>
                            </div>
                            {/* Key Metrics */}
                            <div className="flex flex-col space-y-2 text-md text-gray-700">
                                <span className="flex items-center space-x-2">
                                    <Activity className="w-5 h-5 text-red-500" />
                                    <span>P(Completion): <span className="font-bold text-lg">{recommendation.pCompleteRecommended}</span></span>
                                </span>
                                <span className="flex items-center space-x-2">
                                    <TrendingUp className="w-5 h-5 text-orange-500" />
                                    <span>Gross Margin (%): <span className="font-bold text-lg">{recommendation.confidence}%</span></span>
                                </span>
                            </div>
                        </div>

                        {/* Price Context Comparison - NEW SECTION */}
                        <div className="mt-6 pt-4">
                            <h4 className="text-xl font-bold mb-4 text-blue-700 flex items-center">
                                <BarChart3 className="w-5 h-5 mr-2" /> Price Comparison
                            </h4>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <PriceComparisonBadge 
                                    label="Historical Cost" 
                                    referencePrice={formData.historicalCost} 
                                    recommendedPrice={recommendation.priceRecommended} 
                                    isCompetitor={false} 
                                />
                                <PriceComparisonBadge 
                                    label="Competitor Price" 
                                    referencePrice={formData.competitorPrice} 
                                    recommendedPrice={recommendation.priceRecommended} 
                                    isCompetitor={true} 
                                />
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Analytics Dashboard Section */}
            {showDashboard && (
                <div className="w-full max-w-5xl mt-6 p-8 bg-white rounded-3xl shadow-2xl border border-gray-100">
                    <h2 className="text-3xl font-extrabold mb-6 text-gray-900 border-b pb-4 border-blue-100 flex items-center">
                        <BarChart3 className="w-7 h-7 mr-3 text-red-600" /> Policy Analytics Dashboard
                    </h2>
                    <AnalyticsDashboard data={dashboardData} />
                </div>
            )}
        </div>
    );
};

export default App;
