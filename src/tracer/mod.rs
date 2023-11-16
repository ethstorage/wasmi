use std::collections::{HashMap, BTreeMap};

use specs::{
    brtable::{ElemEntry, ElemTable},
    configure_table::ConfigureTable,
    etable::{EventTable, EventTableEntry},
    host_function::{HostFunctionDesc, HostPlugin},
    imtable::{InitMemoryTable, InitMemoryTableEntry},
    itable::{InstructionTable, InstructionTableEntry, Opcode},
    jtable::{JumpTable, StaticFrameEntry},
    mtable::{VarType, LocationType, AccessType, MemoryTableEntry},
    state::InitializationState,
    step::StepInfo,
    types::FunctionType, external_host_call_table::ExternalHostCallSignature,
};

use crate::{
    func::FuncInstanceInternal,
    runner::{from_value_internal_to_u64_with_typ, ValueInternal},
    FuncRef,
    GlobalRef,
    MemoryRef,
    Module,
    ModuleRef,
    Signature,
};

use self::{imtable::IMTable, phantom::PhantomFunction};

pub mod etable;
pub mod imtable;
pub mod phantom;

#[derive(Debug)]
pub struct FuncDesc {
    pub index_within_jtable: u32,
    pub ftype: FunctionType,
    pub signature: Signature,
}

#[derive(Debug)]
pub struct TracerCompilationTable {
    pub itable: InstructionTable,
    pub imtable: InitMemoryTable,
    pub etable: EventTable,
    pub jtable: JumpTable,
    pub elem_table: ElemTable,
    pub configure_table: ConfigureTable,
    pub static_jtable: Vec<StaticFrameEntry>,
    // initial state related
    pub next_imtable: InitMemoryTable,
    pub prev_state: InitializationState<u32>,
    pub next_state: InitializationState<u32>,
}

struct Callback(Box<dyn FnMut(TracerCompilationTable)>);
use core::fmt::Debug;
impl Debug for Callback {
    fn fmt(&self, _: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        Ok(())
    }
}

#[derive(Debug)]
pub struct Tracer {
    pub itable: InstructionTable,
    pub imtable: IMTable,
    pub etable: EventTable,
    pub jtable: JumpTable,
    pub elem_table: ElemTable,
    pub configure_table: ConfigureTable,
    type_of_func_ref: Vec<(FuncRef, u32)>,
    function_lookup: Vec<(FuncRef, u32)>,
    pub(crate) last_jump_eid: Vec<u32>,
    function_index_allocator: u32,
    pub(crate) function_index_translation: HashMap<u32, FuncDesc>,
    pub host_function_index_lookup: HashMap<usize, HostFunctionDesc>,
    pub static_jtable_entries: Vec<StaticFrameEntry>,
    pub phantom_functions: Vec<String>,
    pub phantom_functions_ref: Vec<FuncRef>,
    // Wasm Image Function Idx
    pub wasm_input_func_idx: Option<u32>,
    pub wasm_input_func_ref: Option<FuncRef>,
    // perf opt
    pub itable_entries: HashMap<u64, InstructionTableEntry>,
    pub function_map: HashMap<usize, u32>,
    pub host_function_map: HashMap<usize, u32>,
    // continuation related
    pub prev_state: InitializationState<u32>,
    pub next_state: InitializationState<u32>,
    pub cur_imtable: InitMemoryTable,
    local_map: BTreeMap<u32, InitMemoryTableEntry>,
    global_map: BTreeMap<u32, InitMemoryTableEntry>,
    memory_map: BTreeMap<u32, InitMemoryTableEntry>,
    prev_eid: u32,
    callback: Callback,
}

impl Tracer {
    /// Create an empty tracer
    pub fn new(
        host_plugin_lookup: HashMap<usize, HostFunctionDesc>,
        phantom_functions: &Vec<String>,
        callback: impl FnMut(TracerCompilationTable) + 'static,
    ) -> Self {
        Tracer {
            itable: InstructionTable::default(),
            imtable: IMTable::default(),
            etable: EventTable::default(),
            last_jump_eid: vec![],
            jtable: JumpTable::default(),
            elem_table: ElemTable::default(),
            configure_table: ConfigureTable::default(),
            type_of_func_ref: vec![],
            function_lookup: vec![],
            function_index_allocator: 1,
            function_index_translation: Default::default(),
            host_function_index_lookup: host_plugin_lookup,
            static_jtable_entries: vec![],
            phantom_functions: phantom_functions.clone(),
            phantom_functions_ref: vec![],
            wasm_input_func_ref: None,
            wasm_input_func_idx: None,
            itable_entries: HashMap::new(),
            function_map: HashMap::new(),
            host_function_map: HashMap::new(),
            // continuation related
            prev_state: InitializationState::<u32>::default(),
            next_state: InitializationState::<u32>::default(),
            cur_imtable: InitMemoryTable::default(),
            local_map: BTreeMap::<u32, InitMemoryTableEntry>::new(),
            global_map: BTreeMap::<u32, InitMemoryTableEntry>::new(),
            memory_map: BTreeMap::<u32, InitMemoryTableEntry>::new(),
            prev_eid: 0,
            callback: Callback(Box::new(callback)),
        }
    }

    pub fn push_frame(&mut self) {
        self.last_jump_eid.push(self.etable.get_latest_eid());
    }

    pub fn pop_frame(&mut self) {
        self.last_jump_eid.pop().unwrap();
    }

    pub fn last_jump_eid(&self) -> u32 {
        *self.last_jump_eid.last().unwrap()
    }

    pub fn eid(&self) -> u32 {
        self.etable.get_latest_eid()
    }

    pub fn prev_eid(&self) -> u32 {
        self.prev_eid
    }

    fn allocate_func_index(&mut self) -> u32 {
        let r = self.function_index_allocator;
        self.function_index_allocator = r + 1;
        r
    }

    fn lookup_host_plugin(&self, function_index: usize) -> HostFunctionDesc {
        self.host_function_index_lookup
            .get(&function_index)
            .unwrap()
            .clone()
    }
}

impl Tracer {
    fn update_initialization_state(&mut self, is_last_slice: bool) {
        let mut host_public_inputs = self.prev_state.host_public_inputs;
        let mut context_in_index = self.prev_state.context_in_index;
        let mut context_out_index = self.prev_state.context_out_index;
        let mut external_host_call_call_index = self.prev_state.external_host_call_call_index;

        #[cfg(feature = "continuation")]
        let mut jops = self.prev_state.jops;

        for entry in self.etable.entries() {
            match &entry.step_info {
                // TODO: fix hard code
                StepInfo::CallHost {
                    function_name,
                    args,
                    op_index_in_plugin,
                    ..
                } => {
                    if *op_index_in_plugin == HostPlugin::HostInput as usize {
                        if function_name == "wasm_input" && args[0] != 0
                            || function_name == "wasm_output"
                        {
                            host_public_inputs += 1;
                        }
                    } else if *op_index_in_plugin == HostPlugin::Context as usize {
                        if function_name == "wasm_read_context" {
                            context_in_index += 1;
                        } else if function_name == "wasm_write_context" {
                            context_out_index += 1;
                        }
                    }
                }
                StepInfo::ExternalHostCall { .. } => external_host_call_call_index += 1,
                StepInfo::Call { .. } | StepInfo::CallIndirect { .. } | StepInfo::Return { .. } => {
                    #[cfg(feature = "continuation")]
                    {
                        jops += 1;
                    }
                }
                _ => (),
            }
        }

        let last_entry = self.etable.entries().last().unwrap();

        let post_initialization_state = if is_last_slice {
            InitializationState {
                eid: last_entry.eid + 1,
                fid: 0,
                iid: 0,
                frame_id: 0,
                // TODO: why not constant 4095?
                sp: last_entry.sp
                    + if let Opcode::Return { drop, .. } = last_entry.inst.opcode {
                        drop
                    } else {
                        0
                    },

                host_public_inputs,
                context_in_index,
                context_out_index,
                external_host_call_call_index,

                initial_memory_pages: last_entry.allocated_memory_pages,
                maximal_memory_pages: self.configure_table.maximal_memory_pages,

                #[cfg(feature = "continuation")]
                jops,
            }
        } else {
            InitializationState {
                eid: last_entry.eid,
                fid: last_entry.inst.fid,
                iid: last_entry.inst.iid,
                frame_id: last_entry.last_jump_eid,
                // TODO: why not constant 4095?
                sp: last_entry.sp,

                host_public_inputs,
                context_in_index,
                context_out_index,
                external_host_call_call_index,

                initial_memory_pages: last_entry.allocated_memory_pages,
                maximal_memory_pages: self.configure_table.maximal_memory_pages,

                #[cfg(feature = "continuation")]
                jops,
            }
        };

        self.next_state = post_initialization_state;
    }

    pub(crate) fn update_init_memory_map(&mut self) {
        let imtable = self.imtable.finalized(); // there lies a costly merge sort in this function
        let mut entries = imtable.entries().clone();

        for i in 0..entries.len() {
            let entry = entries.pop().unwrap();
            match entry.ltype {
                LocationType::Stack => {
                    assert_eq!(entry.start_offset, entry.end_offset);

                    self.local_map.insert(entry.start_offset, entry);
                }
                LocationType::Heap => {
                    for offset in entry.start_offset..=entry.end_offset {
                        self.memory_map.insert(
                            offset,
                            InitMemoryTableEntry {
                                ltype: entry.ltype,
                                is_mutable: entry.is_mutable,
                                start_offset: offset,
                                end_offset: offset,
                                vtype: entry.vtype,
                                value: entry.value,
                                eid: entry.eid,
                            },
                        );
                    }
                }
                LocationType::Global => {
                    assert_eq!(entry.start_offset, entry.end_offset);

                    self.global_map.insert(entry.start_offset, entry);
                }
            }
        }

        println!("inserted...");
        let init_memory_entries = vec![]
            .into_iter()
            .chain(self.local_map.clone().into_values())
            .chain(self.global_map.clone().into_values())
            .chain(self.memory_map.clone().into_values())
            .collect();
        self.cur_imtable = InitMemoryTable::new(init_memory_entries);
    }

    fn update_cur_memory_table(&mut self) {
        for etable_entry in self.etable.entries() {
            let memory_writing_entires = memory_event_of_step(etable_entry)
                .into_iter()
                .filter(|entry| entry.atype == AccessType::Write);

            for mentry in memory_writing_entires {
                let map = match mentry.ltype {
                    LocationType::Stack => &mut self.local_map,
                    LocationType::Heap => &mut self.memory_map,
                    LocationType::Global => &mut self.global_map,
                };

                map.insert(
                    mentry.offset,
                    InitMemoryTableEntry {
                        ltype: mentry.ltype,
                        is_mutable: mentry.is_mutable,
                        start_offset: mentry.offset,
                        end_offset: mentry.offset,
                        vtype: mentry.vtype,
                        value: mentry.value,
                        eid: etable_entry.eid,
                    },
                );
            }
        }

        let init_memory_entries = vec![]
            .into_iter()
            .chain(self.local_map.clone().into_values())
            .chain(self.global_map.clone().into_values())
            .chain(self.memory_map.clone().into_values())
            .collect();

        self.cur_imtable = InitMemoryTable::new(init_memory_entries);
    }

    pub(crate) fn invoke_callback(&mut self, is_last_slice: bool) {
        // update next_state first
        self.update_initialization_state(is_last_slice);
        self.update_cur_memory_table();

        self.prev_eid = self.eid();
        let etable = std::mem::take(&mut self.etable);
        self.etable.latest_eid = self.prev_eid;
        let tables = TracerCompilationTable {
            itable: self.itable.clone(),
            // imtable: self.imtable.finalized(),
            imtable: InitMemoryTable::default(),
            etable,
            jtable: self.jtable.clone(),
            elem_table: self.elem_table.clone(),
            configure_table: self.configure_table.clone(),
            static_jtable: self.static_jtable_entries.clone(),
            // initial state related
            // next_imtable: self.next_imtable.finalized(),
            next_imtable: InitMemoryTable::default(),
            prev_state: self.prev_state.clone(),
            next_state: self.next_state.clone(),
        };

        // update prev state to current
        _ = std::mem::replace(&mut self.prev_state, self.next_state.clone());

        self.callback.0(tables)
    }
}

impl Tracer {
    fn push_init_memory_intable(table: &mut IMTable, memref: MemoryRef) {
        let pages = (*memref).limits().initial();
        // one page contains 64KB*1024/8=8192 u64 entries
        for i in 0..(pages * 8192) {
            let mut buf = [0u8; 8];
            (*memref).get_into(i * 8, &mut buf).unwrap();
            table.push(false, true, i, i, VarType::I64, u64::from_le_bytes(buf));
        }

        table.push(
            false,
            true,
            pages * 8192,
            memref
                .limits()
                .maximum()
                .map(|limit| limit * 8192 - 1)
                .unwrap_or(u32::MAX),
            VarType::I64,
            0,
        );
    }

    pub(crate) fn push_init_memory(&mut self, memref: MemoryRef) {
        Tracer::push_init_memory_intable(&mut self.imtable, memref);
    }

    // pub(crate) fn push_next_memory(&mut self, memref: MemoryRef) {
    //     // should do a test to check whethere the final execution table is correct
    //     Tracer::push_init_memory_intable(&mut self.next_imtable, memref);
    // }

    fn push_global_intable(table: &mut IMTable, globalidx: u32, globalref: &GlobalRef) {
        let vtype = globalref.elements_value_type().into();

        table.push(
            true,
            globalref.is_mutable(),
            globalidx,
            globalidx,
            vtype,
            from_value_internal_to_u64_with_typ(vtype, ValueInternal::from(globalref.get())),
        );
    }

    pub(crate) fn push_global(&mut self, globalidx: u32, globalref: &GlobalRef) {
        Tracer::push_global_intable(&mut self.imtable, globalidx, globalref);
    }

    // pub(crate) fn push_next_global(&mut self, globalidx: u32, globalref: &GlobalRef) {
    //     Tracer::push_global_intable(&mut self.next_imtable, globalidx, globalref);
    // }

    pub(crate) fn push_elem(&mut self, table_idx: u32, offset: u32, func_idx: u32, type_idx: u32) {
        self.elem_table.insert(ElemEntry {
            table_idx,
            type_idx,
            offset,
            func_idx,
        })
    }

    pub(crate) fn push_type_of_func_ref(&mut self, func: FuncRef, type_idx: u32) {
        self.type_of_func_ref.push((func, type_idx))
    }

    #[allow(dead_code)]
    pub(crate) fn statistics_instructions<'a>(&mut self, module_instance: &ModuleRef) {
        let mut func_index = 0;
        let mut insts = vec![];

        loop {
            if let Some(func) = module_instance.func_by_index(func_index) {
                let body = func.body().unwrap();

                let code = &body.code.vec;

                for inst in code {
                    if insts.iter().position(|i| i == inst).is_none() {
                        insts.push(inst.clone())
                    }
                }
            } else {
                break;
            }

            func_index = func_index + 1;
        }

        for inst in insts {
            println!("{:?}", inst);
        }
    }

    pub(crate) fn lookup_type_of_func_ref(&self, func_ref: &FuncRef) -> u32 {
        self.type_of_func_ref
            .iter()
            .find(|&f| f.0 == *func_ref)
            .unwrap()
            .1
    }

    pub(crate) fn register_module_instance(
        &mut self,
        module: &Module,
        module_instance: &ModuleRef,
    ) {
        let start_fn_idx = module.module().start_section();

        {
            let mut func_index = 0;

            loop {
                if let Some(func) = module_instance.func_by_index(func_index) {
                    if Some(&func) == self.wasm_input_func_ref.as_ref() {
                        self.wasm_input_func_idx = Some(func_index)
                    }

                    let func_index_in_itable = if Some(func_index) == start_fn_idx {
                        0
                    } else {
                        self.allocate_func_index()
                    };

                    let ftype = match *func.as_internal() {
                        crate::func::FuncInstanceInternal::Internal { .. } => {
                            FunctionType::WasmFunction
                        }
                        crate::func::FuncInstanceInternal::Host {
                            host_func_index, ..
                        } => {
                            let plugin_desc = self.lookup_host_plugin(host_func_index);

                            match plugin_desc {
                                HostFunctionDesc::Internal {
                                    name,
                                    op_index_in_plugin,
                                    plugin,
                                } => FunctionType::HostFunction {
                                    plugin,
                                    function_index: host_func_index,
                                    function_name: name,
                                    op_index_in_plugin,
                                },
                                HostFunctionDesc::External { name, op, sig } => {
                                    FunctionType::HostFunctionExternal {
                                        function_name: name,
                                        op,
                                        sig,
                                    }
                                }
                            }
                        }
                    };

                    self.function_lookup
                        .push((func.clone(), func_index_in_itable));

                    match *func.as_internal() {
                        FuncInstanceInternal::Internal {
                            image_func_index, ..
                        } => {
                            self.function_map
                                .insert(image_func_index, func_index_in_itable);
                        }
                        FuncInstanceInternal::Host {
                            host_func_index, ..
                        } => {
                            self.host_function_map
                                .insert(host_func_index, func_index_in_itable);
                        }
                    }

                    self.function_index_translation.insert(
                        func_index,
                        FuncDesc {
                            index_within_jtable: func_index_in_itable,
                            ftype,
                            signature: func.signature().clone(),
                        },
                    );
                    func_index = func_index + 1;
                } else {
                    break;
                }
            }
        }

        {
            let phantom_functions_ref = self.phantom_functions.clone();

            for func_name in phantom_functions_ref {
                let func = module_instance.func_by_name(&func_name).unwrap();

                self.push_phantom_function(func);
            }
        }

        {
            let mut func_index = 0;

            loop {
                if let Some(func) = module_instance.func_by_index(func_index) {
                    let funcdesc = self.function_index_translation.get(&func_index).unwrap();

                    if self.is_phantom_function(&func) {
                        let instructions = PhantomFunction::build_phantom_function_instructions(
                            &funcdesc.signature,
                            self.wasm_input_func_idx.unwrap(),
                        );

                        for (iid, inst) in instructions.into_iter().enumerate() {
                            self.itable.push(
                                funcdesc.index_within_jtable,
                                iid as u32,
                                inst.into(&self.function_index_translation),
                            )
                        }
                    } else {
                        if let Some(body) = func.body() {
                            let code = &body.code;
                            let mut iter = code.iterate_from(0);
                            loop {
                                let pc = iter.position();
                                if let Some(instruction) = iter.next() {
                                    let _ = self.itable.push(
                                        funcdesc.index_within_jtable,
                                        pc,
                                        instruction.into(&self.function_index_translation),
                                    );
                                    self.itable_entries.insert(
                                        ((funcdesc.index_within_jtable as u64) << 32) + pc as u64,
                                        self.itable.entries().last().unwrap().clone(),
                                    );
                                } else {
                                    break;
                                }
                            }
                        }
                    }

                    func_index = func_index + 1;
                } else {
                    break;
                }
            }
        }
    }

    pub fn lookup_function(&self, function: &FuncRef) -> u32 {
        match *function.as_internal() {
            FuncInstanceInternal::Internal {
                image_func_index, ..
            } => *self.function_map.get(&image_func_index).unwrap(),
            FuncInstanceInternal::Host {
                host_func_index, ..
            } => *self.host_function_map.get(&host_func_index).unwrap(),
        }
    }

    pub fn lookup_ientry(&self, function: &FuncRef, pos: u32) -> InstructionTableEntry {
        let function_idx = self.lookup_function(function);
        let key = ((function_idx as u64) << 32) + pos as u64;
        return self.itable_entries.get(&key).unwrap().clone();

        // unreachable!()
    }

    pub fn lookup_first_inst(&self, function: &FuncRef) -> InstructionTableEntry {
        let function_idx = self.lookup_function(function);

        for ientry in self.itable.entries() {
            if ientry.fid == function_idx {
                return ientry.clone();
            }
        }

        unreachable!();
    }

    pub fn push_phantom_function(&mut self, function: FuncRef) {
        self.phantom_functions_ref.push(function)
    }

    pub fn is_phantom_function(&self, func: &FuncRef) -> bool {
        self.phantom_functions_ref.contains(func)
    }
}

pub fn memory_event_of_step(event: &EventTableEntry) -> Vec<MemoryTableEntry> {
    let eid = event.eid;
    let sp_before_execution = event.sp;

    match &event.step_info {
        StepInfo::Br {
            drop,
            keep,
            keep_values,
            ..
        } => {
            assert_eq!(keep.len(), keep_values.len());
            assert!(keep.len() <= 1);

            let mut sp = sp_before_execution + 1;
            let mut ops = vec![];

            {
                for i in 0..keep.len() {
                    ops.push(MemoryTableEntry {
                        eid,
                        offset: sp,
                        ltype: LocationType::Stack,
                        atype: AccessType::Read,
                        vtype: keep[i].into(),
                        is_mutable: true,
                        value: keep_values[i],
                    });

                    sp = sp + 1;
                }
            }

            sp += drop;
            sp -= 1;

            {
                for i in 0..keep.len() {
                    ops.push(MemoryTableEntry {
                        eid,
                        offset: sp,
                        ltype: LocationType::Stack,
                        atype: AccessType::Write,
                        vtype: keep[i].into(),
                        is_mutable: true,
                        value: keep_values[i],
                    });

                    sp = sp - 1;
                }
            }

            ops
        }
        StepInfo::BrIfEqz {
            condition,
            drop,
            keep,
            keep_values,
            ..
        } => {
            assert_eq!(keep.len(), keep_values.len());
            assert!(keep.len() <= 1);

            let mut sp = sp_before_execution + 1;

            let mut ops = vec![MemoryTableEntry {
                eid,
                offset: sp,
                ltype: LocationType::Stack,
                atype: AccessType::Read,
                vtype: VarType::I32,
                is_mutable: true,
                value: *condition as u32 as u64,
            }];

            sp = sp + 1;

            if *condition != 0 {
                return ops;
            }

            {
                for i in 0..keep.len() {
                    ops.push(MemoryTableEntry {
                        eid,
                        offset: sp,
                        ltype: LocationType::Stack,
                        atype: AccessType::Read,
                        vtype: keep[i].into(),
                        is_mutable: true,
                        value: keep_values[i],
                    });

                    sp = sp + 1;
                }
            }

            sp += drop;
            sp -= 1;

            {
                for i in 0..keep.len() {
                    ops.push(MemoryTableEntry {
                        eid,
                        offset: sp,
                        ltype: LocationType::Stack,
                        atype: AccessType::Write,
                        vtype: keep[i].into(),
                        is_mutable: true,
                        value: keep_values[i],
                    });

                    sp = sp - 1;
                }
            }

            ops
        }
        StepInfo::BrIfNez {
            condition,
            drop,
            keep,
            keep_values,
            ..
        } => {
            assert_eq!(keep.len(), keep_values.len());
            assert!(keep.len() <= 1);

            let mut sp = sp_before_execution + 1;

            let mut ops = vec![MemoryTableEntry {
                eid,
                offset: sp,
                ltype: LocationType::Stack,
                atype: AccessType::Read,
                vtype: VarType::I32,
                is_mutable: true,
                value: *condition as u32 as u64,
            }];

            sp = sp + 1;

            if *condition == 0 {
                return ops;
            }

            {
                for i in 0..keep.len() {
                    ops.push(MemoryTableEntry {
                        eid,
                        offset: sp,
                        ltype: LocationType::Stack,
                        atype: AccessType::Read,
                        vtype: keep[i].into(),
                        is_mutable: true,
                        value: keep_values[i],
                    });

                    sp = sp + 1;
                }
            }

            sp += drop;
            sp -= 1;

            {
                for i in 0..keep.len() {
                    ops.push(MemoryTableEntry {
                        eid,
                        offset: sp,
                        ltype: LocationType::Stack,
                        atype: AccessType::Write,
                        vtype: keep[i].into(),
                        is_mutable: true,
                        value: keep_values[i],
                    });

                    sp = sp - 1;
                }
            }

            ops
        }
        StepInfo::BrTable {
            index,
            drop,
            keep,
            keep_values,
            ..
        } => {
            assert_eq!(keep.len(), keep_values.len());
            assert!(keep.len() <= 1);

            let mut sp = sp_before_execution + 1;

            let mut ops = vec![MemoryTableEntry {
                eid,
                offset: sp,
                ltype: LocationType::Stack,
                atype: AccessType::Read,
                vtype: VarType::I32,
                is_mutable: true,
                value: *index as u32 as u64,
            }];

            sp = sp + 1;

            {
                for i in 0..keep.len() {
                    ops.push(MemoryTableEntry {
                        eid,
                        offset: sp,
                        ltype: LocationType::Stack,
                        atype: AccessType::Read,
                        vtype: keep[i].into(),
                        is_mutable: true,
                        value: keep_values[i],
                    });

                    sp = sp + 1;
                }
            }

            sp += drop;
            sp -= 1;

            {
                for i in 0..keep.len() {
                    ops.push(MemoryTableEntry {
                        eid,
                        offset: sp,
                        ltype: LocationType::Stack,
                        atype: AccessType::Write,
                        vtype: keep[i].into(),
                        is_mutable: true,
                        value: keep_values[i],
                    });

                    sp = sp - 1;
                }
            }

            ops
        }
        StepInfo::Return {
            drop,
            keep,
            keep_values,
        } => {
            assert_eq!(keep.len(), keep_values.len());
            assert!(keep.len() <= 1);

            let mut sp = sp_before_execution + 1;
            let mut ops = vec![];

            {
                for i in 0..keep.len() {
                    ops.push(MemoryTableEntry {
                        eid,
                        offset: sp,
                        ltype: LocationType::Stack,
                        atype: AccessType::Read,
                        vtype: keep[i].into(),
                        is_mutable: true,
                        value: keep_values[i],
                    });

                    sp = sp + 1;
                }
            }

            sp += drop;
            sp -= 1;

            {
                for i in 0..keep.len() {
                    ops.push(MemoryTableEntry {
                        eid,
                        offset: sp,
                        ltype: LocationType::Stack,
                        atype: AccessType::Write,
                        vtype: keep[i].into(),
                        is_mutable: true,
                        value: keep_values[i],
                    });

                    sp = sp - 1;
                }
            }

            ops
        }
        StepInfo::Drop { .. } => vec![],
        StepInfo::Select {
            val1,
            val2,
            cond,
            result,
            vtype,
        } => {
            let mut sp = sp_before_execution + 1;
            let mut ops = vec![];

            ops.push(MemoryTableEntry {
                eid,
                offset: sp,
                ltype: LocationType::Stack,
                atype: AccessType::Read,
                vtype: VarType::I32,
                is_mutable: true,
                value: *cond,
            });
            sp = sp + 1;

            ops.push(MemoryTableEntry {
                eid,
                offset: sp,
                ltype: LocationType::Stack,
                atype: AccessType::Read,
                vtype: *vtype,
                is_mutable: true,
                value: *val2,
            });
            sp = sp + 1;

            ops.push(MemoryTableEntry {
                eid,
                offset: sp,
                ltype: LocationType::Stack,
                atype: AccessType::Read,
                vtype: *vtype,
                is_mutable: true,
                value: *val1,
            });

            ops.push(MemoryTableEntry {
                eid,
                offset: sp,
                ltype: LocationType::Stack,
                atype: AccessType::Write,
                vtype: *vtype,
                is_mutable: true,
                value: *result,
            });

            ops
        }
        StepInfo::Call { index: _ } => {
            vec![]
        }
        StepInfo::CallIndirect { offset, .. } => {
            let stack_read = MemoryTableEntry {
                eid,
                offset: sp_before_execution + 1,
                ltype: LocationType::Stack,
                atype: AccessType::Read,
                vtype: VarType::I32,
                is_mutable: true,
                value: *offset as u64,
            };

            vec![stack_read]
        }
        StepInfo::CallHost {
            args,
            ret_val,
            signature,
            ..
        } => {
            let mut mops = vec![];
            let mut sp = sp_before_execution;

            for (i, (ty, val)) in signature.params.iter().zip(args.iter()).enumerate() {
                mops.push(MemoryTableEntry {
                    eid,
                    offset: sp_before_execution + args.len() as u32 - i as u32,
                    ltype: LocationType::Stack,
                    atype: AccessType::Read,
                    vtype: (*ty).into(),
                    is_mutable: true,
                    value: *val,
                });
            }

            sp = sp + args.len() as u32;

            if let Some(ty) = signature.return_type {
                mops.push(MemoryTableEntry {
                    eid,
                    offset: sp,
                    ltype: LocationType::Stack,
                    atype: AccessType::Write,
                    vtype: ty.into(),
                    is_mutable: true,
                    value: ret_val.unwrap(),
                });
            }

            mops
        }
        StepInfo::ExternalHostCall { value, sig, .. } => match sig {
            ExternalHostCallSignature::Argument => {
                let stack_read = MemoryTableEntry {
                    eid,
                    offset: sp_before_execution + 1,
                    ltype: LocationType::Stack,
                    atype: AccessType::Read,
                    vtype: VarType::I64,
                    is_mutable: true,
                    value: value.unwrap(),
                };

                vec![stack_read]
            }
            ExternalHostCallSignature::Return => {
                let stack_write = MemoryTableEntry {
                    eid,
                    offset: sp_before_execution,
                    ltype: LocationType::Stack,
                    atype: AccessType::Write,
                    vtype: VarType::I64,
                    is_mutable: true,
                    value: value.unwrap(),
                };

                vec![stack_write]
            }
        },

        StepInfo::GetLocal {
            vtype,
            depth,
            value,
        } => {
            let read = MemoryTableEntry {
                eid,
                offset: sp_before_execution + depth,
                ltype: LocationType::Stack,
                atype: AccessType::Read,
                vtype: *vtype,
                is_mutable: true,
                value: *value,
            };

            let write = MemoryTableEntry {
                eid,
                offset: sp_before_execution,
                ltype: LocationType::Stack,
                atype: AccessType::Write,
                vtype: *vtype,
                is_mutable: true,
                value: *value,
            };
            vec![read, write]
        }
        StepInfo::SetLocal {
            vtype,
            depth,
            value,
        } => {
            let mut sp = sp_before_execution;

            let read = MemoryTableEntry {
                eid,
                offset: sp + 1,
                ltype: LocationType::Stack,
                atype: AccessType::Read,
                vtype: *vtype,
                is_mutable: true,
                value: *value,
            };

            sp += 1;

            let write = MemoryTableEntry {
                eid,
                offset: sp + depth,
                ltype: LocationType::Stack,
                atype: AccessType::Write,
                vtype: *vtype,
                is_mutable: true,
                value: *value,
            };

            vec![read, write]
        }
        StepInfo::TeeLocal {
            vtype,
            depth,
            value,
        } => {
            let read = MemoryTableEntry {
                eid,
                offset: sp_before_execution + 1,
                ltype: LocationType::Stack,
                atype: AccessType::Read,
                vtype: *vtype,
                is_mutable: true,
                value: *value,
            };

            let write = MemoryTableEntry {
                eid,
                offset: sp_before_execution + depth,
                ltype: LocationType::Stack,
                atype: AccessType::Write,
                vtype: *vtype,
                is_mutable: true,
                value: *value,
            };

            vec![read, write]
        }

        StepInfo::GetGlobal {
            idx,
            vtype,
            is_mutable,
            value,
            ..
        } => {
            let global_get = MemoryTableEntry {
                eid,
                offset: *idx,
                ltype: LocationType::Global,
                atype: AccessType::Read,
                vtype: *vtype,
                is_mutable: *is_mutable,
                value: *value,
            };

            let stack_write = MemoryTableEntry {
                eid,
                offset: sp_before_execution,
                ltype: LocationType::Stack,
                atype: AccessType::Write,
                vtype: *vtype,
                is_mutable: true,
                value: *value,
            };

            vec![global_get, stack_write]
        }
        StepInfo::SetGlobal {
            idx,
            vtype,
            is_mutable,
            value,
        } => {
            let stack_read = MemoryTableEntry {
                eid,
                offset: sp_before_execution + 1,
                ltype: LocationType::Stack,
                atype: AccessType::Read,
                vtype: *vtype,
                is_mutable: true,
                value: *value,
            };

            let global_set = MemoryTableEntry {
                eid,
                offset: *idx,
                ltype: LocationType::Global,
                atype: AccessType::Write,
                vtype: *vtype,
                is_mutable: *is_mutable,
                value: *value,
            };

            vec![stack_read, global_set]
        }

        StepInfo::Load {
            vtype,
            load_size,
            raw_address,
            effective_address,
            value,
            block_value1,
            block_value2,
            ..
        } => {
            let load_address_from_stack = MemoryTableEntry {
                eid,
                offset: sp_before_execution + 1,
                ltype: LocationType::Stack,
                atype: AccessType::Read,
                vtype: VarType::I32,
                is_mutable: true,
                value: *raw_address as u64,
            };

            let load_value1 = MemoryTableEntry {
                eid,
                offset: (*effective_address) / 8,
                ltype: LocationType::Heap,
                atype: AccessType::Read,
                // Load u64 from address which align with 8
                vtype: VarType::I64,
                is_mutable: true,
                // The value will be used to lookup within imtable, hence block_value is given here
                value: *block_value1,
            };

            let load_value2 = if *effective_address % 8 + load_size.byte_size() as u32 > 8 {
                Some(MemoryTableEntry {
                    eid,
                    offset: effective_address / 8 + 1,
                    ltype: LocationType::Heap,
                    atype: AccessType::Read,
                    // Load u64 from address which align with 8
                    vtype: VarType::I64,
                    is_mutable: true,
                    // The value will be used to lookup within imtable, hence block_value is given here
                    value: *block_value2,
                })
            } else {
                None
            };

            let push_value = MemoryTableEntry {
                eid,
                offset: sp_before_execution + 1,
                ltype: LocationType::Stack,
                atype: AccessType::Write,
                vtype: *vtype,
                is_mutable: true,
                value: *value,
            };

            vec![
                vec![load_address_from_stack, load_value1],
                load_value2.map_or(vec![], |v| vec![v]),
                vec![push_value],
            ]
            .concat()
        }
        StepInfo::Store {
            vtype,
            store_size,
            raw_address,
            effective_address,
            value,
            pre_block_value1,
            updated_block_value1,
            pre_block_value2,
            updated_block_value2,
            ..
        } => {
            let load_value_from_stack = MemoryTableEntry {
                eid,
                offset: sp_before_execution + 1,
                ltype: LocationType::Stack,
                atype: AccessType::Read,
                vtype: *vtype,
                is_mutable: true,
                value: *value,
            };

            let load_address_from_stack = MemoryTableEntry {
                eid,
                offset: sp_before_execution + 2,
                ltype: LocationType::Stack,
                atype: AccessType::Read,
                vtype: VarType::I32,
                is_mutable: true,
                value: *raw_address as u64,
            };

            let load_value1 = MemoryTableEntry {
                eid,
                offset: effective_address / 8,
                ltype: LocationType::Heap,
                atype: AccessType::Read,
                // Load u64 from address which align with 8
                vtype: VarType::I64,
                is_mutable: true,
                // The value will be used to lookup within imtable, hence block_value is given here
                value: *pre_block_value1,
            };

            let write_value1 = MemoryTableEntry {
                eid,
                offset: effective_address / 8,
                ltype: LocationType::Heap,
                atype: AccessType::Write,
                // Load u64 from address which align with 8
                vtype: VarType::I64,
                is_mutable: true,
                // The value will be used to lookup within imtable, hence block_value is given here
                value: *updated_block_value1,
            };

            if *effective_address % 8 + store_size.byte_size() as u32 > 8 {
                let load_value2 = MemoryTableEntry {
                    eid,
                    offset: effective_address / 8 + 1,
                    ltype: LocationType::Heap,
                    atype: AccessType::Read,
                    // Load u64 from address which align with 8
                    vtype: VarType::I64,
                    is_mutable: true,
                    // The value will be used to lookup within imtable, hence block_value is given here
                    value: *pre_block_value2,
                };

                let write_value2 = MemoryTableEntry {
                    eid,
                    offset: effective_address / 8 + 1,
                    ltype: LocationType::Heap,
                    atype: AccessType::Write,
                    // Load u64 from address which align with 8
                    vtype: VarType::I64,
                    is_mutable: true,
                    // The value will be used to lookup within imtable, hence block_value is given here
                    value: *updated_block_value2,
                };
                vec![
                    load_value_from_stack,
                    load_address_from_stack,
                    load_value1,
                    write_value1,
                    load_value2,
                    write_value2,
                ]
            } else {
                vec![
                    load_value_from_stack,
                    load_address_from_stack,
                    load_value1,
                    write_value1,
                ]
            }
        }

        StepInfo::MemorySize => mem_op_from_stack_only_step(
            sp_before_execution,
            eid,
            VarType::I32,
            VarType::I32,
            &[],
            &[event.allocated_memory_pages as u32 as u64],
        ),
        StepInfo::MemoryGrow { grow_size, result } => mem_op_from_stack_only_step(
            sp_before_execution,
            eid,
            VarType::I32,
            VarType::I32,
            &[*grow_size as u32 as u64],
            &[*result as u32 as u64],
        ),

        StepInfo::I32Const { value } => mem_op_from_stack_only_step(
            sp_before_execution,
            eid,
            VarType::I32,
            VarType::I32,
            &[],
            &[*value as u32 as u64],
        ),
        StepInfo::I32BinOp {
            left, right, value, ..
        }
        | StepInfo::I32BinShiftOp {
            left, right, value, ..
        }
        | StepInfo::I32BinBitOp {
            left, right, value, ..
        } => mem_op_from_stack_only_step(
            sp_before_execution,
            eid,
            VarType::I32,
            VarType::I32,
            &[*right as u32 as u64, *left as u32 as u64],
            &[*value as u32 as u64],
        ),
        StepInfo::I32Comp {
            left, right, value, ..
        } => mem_op_from_stack_only_step(
            sp_before_execution,
            eid,
            VarType::I32,
            VarType::I32,
            &[*right as u32 as u64, *left as u32 as u64],
            &[*value as u32 as u64],
        ),

        StepInfo::I64BinOp {
            left, right, value, ..
        }
        | StepInfo::I64BinShiftOp {
            left, right, value, ..
        }
        | StepInfo::I64BinBitOp {
            left, right, value, ..
        } => mem_op_from_stack_only_step(
            sp_before_execution,
            eid,
            VarType::I64,
            VarType::I64,
            &[*right as u64, *left as u64],
            &[*value as u64],
        ),

        StepInfo::I64Const { value } => mem_op_from_stack_only_step(
            sp_before_execution,
            eid,
            VarType::I64,
            VarType::I64,
            &[],
            &[*value as u64],
        ),
        StepInfo::I64Comp {
            left, right, value, ..
        } => mem_op_from_stack_only_step(
            sp_before_execution,
            eid,
            VarType::I64,
            VarType::I32,
            &[*right as u64, *left as u64],
            &[*value as u32 as u64],
        ),
        StepInfo::UnaryOp {
            vtype,
            operand,
            result,
            ..
        } => mem_op_from_stack_only_step(
            sp_before_execution,
            eid,
            *vtype,
            *vtype,
            &[*operand],
            &[*result],
        ),

        StepInfo::Test {
            vtype,
            value,
            result,
        } => mem_op_from_stack_only_step(
            sp_before_execution,
            eid,
            *vtype,
            VarType::I32,
            &[*value],
            &[*result as u32 as u64],
        ),

        StepInfo::I32WrapI64 { value, result } => mem_op_from_stack_only_step(
            sp_before_execution,
            eid,
            VarType::I64,
            VarType::I32,
            &[*value as u64],
            &[*result as u32 as u64],
        ),
        StepInfo::I64ExtendI32 { value, result, .. } => mem_op_from_stack_only_step(
            sp_before_execution,
            eid,
            VarType::I32,
            VarType::I64,
            &[*value as u32 as u64],
            &[*result as u64],
        ),
        StepInfo::I32SignExtendI8 { value, result }
        | StepInfo::I32SignExtendI16 { value, result } => mem_op_from_stack_only_step(
            sp_before_execution,
            eid,
            VarType::I32,
            VarType::I32,
            &[*value as u32 as u64],
            &[*result as u32 as u64],
        ),
        StepInfo::I64SignExtendI8 { value, result }
        | StepInfo::I64SignExtendI16 { value, result }
        | StepInfo::I64SignExtendI32 { value, result } => mem_op_from_stack_only_step(
            sp_before_execution,
            eid,
            VarType::I64,
            VarType::I64,
            &[*value as u64],
            &[*result as u64],
        ),
    }
}

pub(crate) fn mem_op_from_stack_only_step(
    sp_before_execution: u32,
    eid: u32,
    inputs_type: VarType,
    outputs_type: VarType,
    pop_value: &[u64],
    push_value: &[u64],
) -> Vec<MemoryTableEntry> {
    let mut mem_op = vec![];
    let mut sp = sp_before_execution;

    for i in 0..pop_value.len() {
        mem_op.push(MemoryTableEntry {
            eid,
            offset: sp + 1,
            ltype: LocationType::Stack,
            atype: AccessType::Read,
            vtype: inputs_type,
            is_mutable: true,
            value: pop_value[i],
        });
        sp = sp + 1;
    }

    for i in 0..push_value.len() {
        mem_op.push(MemoryTableEntry {
            eid,
            offset: sp,
            ltype: LocationType::Stack,
            atype: AccessType::Write,
            vtype: outputs_type,
            is_mutable: true,
            value: push_value[i],
        });
        sp = sp - 1;
    }

    mem_op
}
