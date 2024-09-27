import SwiftUI
import CoreML

struct ContentView: View {
    @State private var celsius: String = ""
    @State private var fahrenheit: String = "Fahrenheit value will appear here"

    var body: some View {
        VStack {
            TextField("Enter Celsius", text: $celsius)
                .keyboardType(.decimalPad)
                .padding()
                .textFieldStyle(RoundedBorderTextFieldStyle())

            Button("Get Fahrenheit value ") {
                if let celciusValue = Double(celsius) {
                    fahrenheit = infarFahrenheit(celsius: celciusValue)
                }
       
            }
            .padding()

            Text(fahrenheit)
                .padding()
        }
        .padding()
    }

    
    // Function to use the Core ML model to convert Celsius to Fahrenheit
    func infarFahrenheit(celsius: Double) -> String {
        do {
            let model = try CelsiusToFahrenheit(configuration: MLModelConfiguration())
            
            // Create an MLMultiArray to hold the input
            let inputArray = try MLMultiArray(shape: [1, 1], dataType: .float32)
            inputArray[0] = NSNumber(value: celsius) // Set the value

            // Prepare the input for the model
            let input = CelsiusToFahrenheitInput(dense_input: inputArray)
            
            // Get the prediction from the model
            let output = try model.prediction(input: input)
            
            // Check the output properties
            // Use the actual name found in Xcode (e.g., `Identity`)
            let fahrenheit = output.Identity[0].floatValue // Adjust this to the correct property name

            return "\(fahrenheit) °F"
        } catch {
            return "Error converting temperature."
        }
    }
}

#Preview{
    ContentView()
}
