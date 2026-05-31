import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ResumenTab } from "@/components/dashboard/ResumenTab"
import { HidrogramasTab } from "@/components/dashboard/HidrogramasTab"
import { SensibilidadTab } from "@/components/dashboard/SensibilidadTab"
import { CalibracionTab } from "@/components/dashboard/CalibracionTab"
import { PinnTab } from "@/components/dashboard/PinnTab"

export default function App() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="border-b px-6 py-4">
        <h1 className="text-base font-semibold">Saint-Venant 1D / PINN — Resultados</h1>
        <p className="text-xs text-muted-foreground">
          Modelación hidráulica 1D · Sensibilidad global · Calibración OLS · Red neuronal física
        </p>
      </header>

      <main className="flex-1 px-6 pb-10">
        <Tabs defaultValue="resumen">
          <TabsList className="mt-4">
            <TabsTrigger value="resumen">Resumen</TabsTrigger>
            <TabsTrigger value="hidrogramas">Hidrogramas</TabsTrigger>
            <TabsTrigger value="sensibilidad">Sensibilidad</TabsTrigger>
            <TabsTrigger value="calibracion">Calibración</TabsTrigger>
            <TabsTrigger value="pinn">PINN</TabsTrigger>
          </TabsList>

          <TabsContent value="resumen">
            <ResumenTab />
          </TabsContent>
          <TabsContent value="hidrogramas">
            <HidrogramasTab />
          </TabsContent>
          <TabsContent value="sensibilidad">
            <SensibilidadTab />
          </TabsContent>
          <TabsContent value="calibracion">
            <CalibracionTab />
          </TabsContent>
          <TabsContent value="pinn">
            <PinnTab />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}
