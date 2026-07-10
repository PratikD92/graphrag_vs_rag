from ragas import experiment
import asyncio


@experiment()
async def experiment_compare_models(row, model_name: str, temperature: float):
    # Configure your system with the parameters
    response = await my_system_function(
        row["input"], model=model_name, temperature=temperature
    )

    return {
        **row,
        "response": response,
        "experiment_name": f"{model_name}_temp_{temperature}",
        "model_name": model_name,
        "temperature": temperature,
    }
