package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"
	ort "github.com/yalue/onnxruntime_go"
)

type PreprocessingParams struct {
	NumericalFeatures []string  `json:"numerical_features"`
	LogFeatures       []string  `json:"log_features"`
	ScalerMean        []float64 `json:"scaler_mean"`
	ScalerScale       []float64 `json:"scaler_scale"`
	ImputerStatistics []float64 `json:"imputer_statistics"`
}

var params PreprocessingParams
var ortSession *ort.DynamicAdvancedSession

type FinancialRequest struct {
	Revenue              float64 `json:"Revenue" binding:"required"`
	NetIncome            float64 `json:"Net Income" binding:"required"`
	TotalAssets          float64 `json:"Total Assets" binding:"required"`
	Equity               float64 `json:"Equity" binding:"required"`
	MarketCapitalization float64 `json:"Market Capitalization" binding:"required"`
	EBITDAMargin         float64 `json:"EBITDA_Margin" binding:"required"`
	ROA                  float64 `json:"ROA" binding:"required"`
	ROE                  float64 `json:"ROE" binding:"required"`
	CashFlowToDebt       float64 `json:"CashFlow_to_Debt" binding:"required"`
	DSCR                 float64 `json:"DSCR" binding:"required"`
	FCF                  float64 `json:"FCF" binding:"required"`
	InventoryTurnover    float64 `json:"Inventory_Turnover" binding:"required"`
	DSO                  float64 `json:"DSO" binding:"required"`
	PBRatio              float64 `json:"P_B_Ratio" binding:"required"`
}

func initONNX() error {
	ort.SetSharedLibraryPath("./libonnxruntime.so")
	err := ort.InitializeEnvironment()
	if err != nil {
		return fmt.Errorf("failed to initialize ONNX environment: %w", err)
	}

	ortSession, err = ort.NewDynamicAdvancedSession("../models/financial_model.onnx", []string{"float_input"}, []string{"label"}, nil)
	if err != nil {
		return fmt.Errorf("failed to create session: %w", err)
	}
	return nil
}

func loadPreprocessing(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, &params)
}

func preprocess(req FinancialRequest) []float32 {
	rawValues := []float64{
		req.Revenue, req.NetIncome, req.TotalAssets, req.Equity, req.MarketCapitalization,
		req.EBITDAMargin, req.ROA, req.ROE, req.CashFlowToDebt, req.DSCR, req.FCF,
		req.InventoryTurnover, req.DSO, req.PBRatio,
	}

	result := make([]float32, 14)

	for i, feat := range params.NumericalFeatures {
		val := rawValues[i]

		for _, logFeat := range params.LogFeatures {
			if feat == logFeat {
				val = math.Log1p(val)
				break
			}
		}

		val = (val - params.ScalerMean[i]) / params.ScalerScale[i]

		if math.IsNaN(val) {
			val = params.ImputerStatistics[i]
		}

		result[i] = float32(val)
	}

	return result
}

func predictHandler(c *gin.Context) {
	var req FinancialRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	inputData := preprocess(req)

	inputTensor, err := ort.NewTensor(ort.NewShape(1, 14), inputData)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create input tensor"})
		return
	}
	defer inputTensor.Destroy()

	outputData := make([]int64, 1)
	outputTensor, err := ort.NewTensor(ort.NewShape(1), outputData)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create output tensor"})
		return
	}
	defer outputTensor.Destroy()

	err = ortSession.Run([]ort.Value{inputTensor}, []ort.Value{outputTensor})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Prediction failed: %v", err)})
		return
	}

	c.JSON(http.StatusOK, gin.H{"prediction_class": outputData[0]})
}

func main() {
	if err := loadPreprocessing("../models/preprocessing.json"); err != nil {
		log.Fatalf("Error loading preprocessing params: %v", err)
	}

	if err := initONNX(); err != nil {
		log.Fatalf("Error initializing ONNX runtime: %v", err)
	}
	defer ort.DestroyEnvironment()
	defer ortSession.Destroy()

	r := gin.Default()
	r.POST("/predict", predictHandler)
	
	log.Println("Starting Go API on :8080...")
	r.Run(":8080")
}
