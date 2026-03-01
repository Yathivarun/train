import { createContext, useState } from "react"

export const ScenarioContext = createContext<any>({
  scenario: "Peak",
  setScenario: () => {}
})

export function ScenarioProvider({ children }: any) {

  const [scenario, setScenario] = useState("Peak")

  return (
    <ScenarioContext.Provider value={{ scenario, setScenario }}>
      {children}
    </ScenarioContext.Provider>
  )
}